"""Read-only single-GPU evaluation of a pinned CTP DCP checkpoint."""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed.checkpoint as dcp
from eval_metrics import frame_metrics, select_frames, summarize

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.ctp_loss import conditional_trajectory_peak_loss
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


def main():
    """Evaluate held-out observations without changing the live training process."""
    """Evaluate all held-out frames without modifying the live training process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-episode", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--training-run-id", default="afcg8d7v")
    parser.add_argument("--release-snapshot-after-load", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(0.32)
    config = PI05Config.from_pretrained(args.snapshot)
    config.device = "cpu"
    config.gradient_checkpointing = False
    config.pretrained_path = None
    policy = PI05Policy(config)
    state = {"model": policy.state_dict()}
    dcp.load(state, checkpoint_id=args.snapshot / "pytorch_model_fsdp_0")
    policy.load_state_dict(state["model"], strict=True)
    del state
    # Match BF16 linear/embedding compute while retaining FP32 norms and other parameters.
    for module in policy.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.to(dtype=torch.bfloat16)
    policy.to("cuda").eval()
    config.device = "cuda"
    train_cfg = json.loads((args.snapshot / "train_config.json").read_text())
    if args.training_run_id != train_cfg["wandb"]["run_id"]:
        raise ValueError("Evaluation run identity differs from checkpoint")
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=train_cfg["dataset"]["repo_id"],
            root=train_cfg["dataset"]["root"],
            episodes=list(range(151, 168)),
            video_backend="pyav",
            eval_split=0,
        ),
        policy=config,
        tolerance_s=0.01,
    )
    dataset = make_dataset(cfg)
    ep = dataset.hf_dataset.data.column("episode_index").to_numpy()
    frames = dataset.hf_dataset.data.column("frame_index").to_numpy()
    assert set(ep.tolist()) == set(range(151, 168)) and len(ep) == 12883
    lengths = {int(e): int(frames[ep == e].max()) + 1 for e in np.unique(ep)}
    indices = select_frames(ep, frames, args.per_episode)
    if args.pilot:
        indices = [int(np.flatnonzero(ep == e)[j]) for e in np.unique(ep) for j in [0, -1]]
    assert len(set(indices)) == len(indices)
    coverage = {}
    for e in range(151, 168):
        selected = [int(frames[i]) for i in indices if ep[i] == e]
        assert min(selected) == 0 and max(selected) == lengths[e] - 1
        coverage[str(e)] = {
            "selected": len(selected),
            "available": lengths[e],
            "first": min(selected),
            "last": max(selected),
            "quarters": sorted({min(3, f * 4 // lengths[e]) for f in selected}),
        }
        if not args.pilot:
            assert coverage[str(e)]["quarters"] == [0, 1, 2, 3]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
        multiprocessing_context="spawn",
        persistent_workers=True,
    )
    pre, post = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=args.snapshot,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    checkpoint_step = json.loads((args.snapshot.parent / "source_checkpoint_complete.json").read_text())[
        "step"
    ]
    provenance = {
        "checkpoint_step": checkpoint_step,
        "training_run_id": args.training_run_id,
        "evaluation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "episodes": list(range(151, 168)),
        "available_frames": len(ep),
        "selected_frames": len(indices),
        "coverage": coverage,
        "sampling": "pilot_first_and_last"
        if args.pilot
        else ("all_frames" if not args.per_episode else "evenly_spaced_per_episode"),
        "per_episode": args.per_episode,
        "precision": "BF16 Linear/Embedding, FP32 other weights and metrics, BF16 autocast",
        "semantics": "GT observation at every anchor; top1 50-step action chunk; no closed-loop success claim",
        "normalization": "checkpoint processors; train-only statistics",
        "pad_policy": "masked metrics exclude invalid future actions; legacy_loss is the unmasked comparison objective",
        "joint_order": [
            "left_j1",
            "left_j2",
            "left_j3",
            "left_j4",
            "left_j5",
            "left_j6",
            "left_gripper",
            "right_j1",
            "right_j2",
            "right_j3",
            "right_j4",
            "right_j5",
            "right_j6",
            "right_gripper",
        ],
        "raw_units": "original dataset command units, per dimension; gripper not pooled with revolute joints",
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    if args.release_snapshot_after_load:
        if (
            args.snapshot.name != "model_snapshot"
            or not (args.snapshot.parent / "source_checkpoint_complete.json").exists()
        ):
            raise ValueError("Only evaluator-owned model snapshots can be released")
        shutil.rmtree(args.snapshot)
        print("MODEL_SNAPSHOT_RELEASED_AFTER_LOAD", flush=True)
    rows = []
    started = time.monotonic()
    with (args.output / "frames.jsonl").open("w") as stream, torch.inference_mode():
        for raw in loader:
            raw_target = raw["action"].numpy().copy()
            raw_state = raw["observation.state"].numpy().copy()
            episode_ids = raw["episode_index"].numpy().reshape(-1)
            frame_ids = raw["frame_index"].numpy().reshape(-1)
            valid = ~raw["action_is_pad"].numpy().astype(bool)
            for i, (e, f) in enumerate(zip(episode_ids, frame_ids, strict=True)):
                assert np.array_equal(valid[i], np.arange(config.chunk_size) + f < lengths[int(e)])
            for key in dataset.meta.camera_keys:
                if raw[key].dtype == torch.uint8:
                    raw[key] = raw[key].float() / 255
            batch = pre(raw)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                images, image_masks = policy._preprocess_images(batch)
                trajectories, logits, widths = policy.model.ctp_parameters(
                    images,
                    image_masks,
                    batch["observation.language.tokens"],
                    batch["observation.language.attention_mask"],
                )
            trajectories, logits, widths = trajectories.float(), logits.float(), widths.float()
            probs = torch.softmax(logits, -1)
            top = probs.argmax(-1)
            pred = trajectories[torch.arange(len(top), device="cuda"), top]
            unnormalized = post(pred.clone()).float().cpu().numpy()
            if not rows:
                roundtrip = post(batch["action"].clone()).float().cpu().numpy()
                assert np.allclose(roundtrip, raw_target, atol=1e-5), "Saved normalizer roundtrip mismatch"
            loss, _ = conditional_trajectory_peak_loss(
                trajectories,
                logits,
                widths,
                batch["action"],
                overlap_weight=config.ctp_overlap_weight,
                probability_floor=config.ctp_probability_floor,
                entropy_target_effective_modes=config.ctp_entropy_target_effective_modes,
                entropy_weight=config.ctp_entropy_weight,
                overlap_probability_weighted=config.ctp_overlap_probability_weighted,
                overlap_detach_widths=config.ctp_overlap_detach_widths,
            )
            new_rows = frame_metrics(
                trajectories.cpu().numpy(),
                probs.cpu().numpy(),
                widths.cpu().numpy(),
                batch["action"].float().cpu().numpy(),
                unnormalized,
                raw_target,
                raw_state,
                valid,
            )
            for i, row in enumerate(new_rows):
                e, f = int(episode_ids[i]), int(frame_ids[i])
                row.update(episode=e, frame=f, phase=min(3, f * 4 // lengths[e]), legacy_loss=float(loss[i]))
                assert np.isfinite(row["legacy_loss"])
                stream.write(json.dumps(row) + "\n")
                rows.append(row)
            if len(rows) % 32 == 0 or len(rows) == len(indices):
                stream.flush()
                print(
                    json.dumps(
                        {
                            "frames": len(rows),
                            "total": len(indices),
                            "seconds": round(time.monotonic() - started, 1),
                            "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                        }
                    ),
                    flush=True,
                )
            time.sleep(args.sleep_seconds)
    assert len(rows) == len(indices) and {r["episode"] for r in rows} == set(range(151, 168))
    summary = {
        "provenance": provenance,
        "overall": summarize(rows),
        "per_episode": {str(e): summarize([r for r in rows if r["episode"] == e]) for e in range(151, 168)},
        "per_phase": {
            str(p): summarize([r for r in rows if r["phase"] == p])
            for p in range(4)
            if any(r["phase"] == p for r in rows)
        },
        "wall_seconds": time.monotonic() - started,
    }
    summary["episode_macro_top1_normalized_mse"] = float(
        np.mean([x["top1_normalized_mse"] for x in summary["per_episode"].values()])
    )
    old_subset = [r for r in rows if r["episode"] == 151 and r["frame"] < 128]
    if len(old_subset) == 128:
        summary["legacy_first128_loss"] = float(np.mean([r["legacy_loss"] for r in old_subset]))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    if args.wandb:
        import wandb

        run = wandb.init(
            project="pi05-ctp-pen168",
            entity=os.environ.get("WANDB_ENTITY"),
            id=f"{args.training_run_id}-heldout17-full"
            if not args.per_episode
            else f"{args.training_run_id}-heldout17-strat{args.per_episode}",
            resume="allow",
            name="pi05_ctp_pens168_full_heldout17"
            if not args.per_episode
            else f"pi05_ctp_pens168_heldout17_strat{args.per_episode}",
            job_type="offline-evaluation",
            config={
                k: provenance[k]
                for k in [
                    "training_run_id",
                    "episodes",
                    "available_frames",
                    "sampling",
                    "precision",
                    "semantics",
                    "normalization",
                    "pad_policy",
                    "joint_order",
                    "raw_units",
                ]
            },
            dir=str(args.output),
        )
        run.define_metric("checkpoint_step")
        run.define_metric("heldout/*", step_metric="checkpoint_step")
        run.log(
            {f"heldout/{k}": v for k, v in summary["overall"].items() if isinstance(v, (float, int))}
            | {
                "checkpoint_step": checkpoint_step,
                "heldout/episode_macro_top1_mse": summary["episode_macro_top1_normalized_mse"],
            }
            | {
                f"heldout/episode_{e}/top1_mse": v["top1_normalized_mse"]
                for e, v in summary["per_episode"].items()
            }
            | {
                f"heldout/phase_{p}/top1_mse": v["top1_normalized_mse"]
                for p, v in summary["per_phase"].items()
            }
            | {f"heldout/joint_{i}/rmse": v for i, v in enumerate(summary["overall"]["raw_joint_rmse"])}
            | {
                f"heldout/mode_{i}/selection_fraction": v / len(rows)
                for i, v in enumerate(summary["overall"]["top1_mode_counts"])
            }
            | {
                f"heldout/mode_{i}/mean_probability": v
                for i, v in enumerate(summary["overall"]["mean_probabilities"])
            }
            | {
                f"heldout/mode_{i}/responsibility_fraction": v
                for i, v in enumerate(summary["overall"]["mean_responsibilities"])
            }
        )
        run.log(
            {
                "per_episode": wandb.Table(
                    columns=["episode", "frames", "top1_mse", "best_of_k_mse", "effective_k"],
                    data=[
                        [
                            e,
                            v["frames"],
                            v["top1_normalized_mse"],
                            v["best_of_k_normalized_mse"],
                            v["conditional_effective_k"],
                        ]
                        for e, v in summary["per_episode"].items()
                    ],
                )
            }
        )
        summary["wandb_url"] = run.url
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        run.finish()
    (args.output / "evaluation_complete.json").write_text(
        json.dumps({"checkpoint_step": checkpoint_step, "frames": len(rows), "episodes": 17})
    )
    print("EVALUATION_COMPLETE", json.dumps(summary["overall"]), flush=True)


if __name__ == "__main__":
    main()
