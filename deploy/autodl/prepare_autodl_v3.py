"""Resolve the full V3 configuration and fresh task-specific BASE processors."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import draccus

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors


def main():
    """Run the explicit task preparation or validation command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--job-name", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    root = args.root
    manifest = json.loads((args.dataset / "preparation_manifest.json").read_text())
    initial = root / "models" / (args.job_name + "_base_init")
    out = root / "configs"
    out.mkdir(parents=True, exist_ok=True)
    initial.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((repo / "deploy/autodl/configs/stabilize.json").read_text())
    cfg["dataset"].update(
        root=str(args.dataset), repo_id=manifest.get("repo_id", "local/pens168_ctp"), eval_split=manifest.get("eval_split", 0.1)
    )
    cfg["policy"].update(
        pretrained_path=str(initial),
        text_tokenizer_name=str(args.tokenizer),
        ctp_split_head=True,
        ctp_detach_width_context=True,
        ctp_optimizer_logits_lr_scale=1.0,
        ctp_optimizer_width_lr_scale=0.5,
        ctp_temperature_initial=20.0,
        ctp_temperature_final=1.0,
        ctp_temperature_anneal_steps=1500,
        ctp_overlap_detach_probabilities=True,
        scheduler_decay_steps=20000,
    )
    cfg.update(
        steps=20000,
        stop_after_step=0,
        resume=False,
        save_freq=2000,
        checkpoint_keep_last=1,
        checkpoint_min_free_gb=10,
        eval_steps=250,
        full_eval_steps=1000,
        eval_samples_per_episode=64,
        max_eval_samples=0,
        ctp_color_jitter=0.1,
        best_checkpoint_metric="ctp_selection_score",
        batch_size=8,
        output_dir=str(root / "outputs" / args.job_name),
        job_name=args.job_name,
    )
    cfg["parallelism"]["dp_shard"] = 4
    cfg["wandb"].update(
        run_id=None,
        resume=None,
        enable=True,
        disable_artifact=True,
        mode="online",
        project="pi05-ctp-pens",
        entity="shenchantian-harbin-institute-of-technology",
        notes="Pen-task V3 from pinned official BASE; full finetuning, split head, five LRs, T20-to1; latest1 complete recovery point plus best model, save every2000. AutoDL pen task, matching bottle V3 training strategy.",
    )
    formal = draccus.decode(TrainPipelineConfig, cfg)
    formal.validate()
    formal.policy.save_pretrained(initial)
    meta = LeRobotDatasetMetadata(manifest.get("repo_id", "local/pens168_ctp"), root=args.dataset)
    pre, post = make_pi05_pre_post_processors(formal.policy, meta.stats)
    pre.save_pretrained(initial)
    post.save_pretrained(initial)
    base = args.base / "model.safetensors"
    link = initial / "model.safetensors"
    if not link.exists():
        os.link(base, link)
    if not link.samefile(base):
        raise RuntimeError("Initialization must share the verified official BASE file")
    target = out / (args.job_name + ".json")
    target.write_text(json.dumps(formal.to_dict(), indent=2) + "\n")
    report = {
        "config": str(target),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "task": manifest["task"],
        "train_episodes": len(manifest["train_episodes"]),
        "validation_episodes": len(manifest["validation_episodes"]),
        "steps": 20000,
        "save_frequency": 2000,
        "keep_latest_full": 1,
        "keep_best_model_only": 1,
    }
    (out / "resolved_configuration.json").write_text(json.dumps(report, indent=2) + "\n")
    print("AUTODL_V3_CONFIG_READY", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
