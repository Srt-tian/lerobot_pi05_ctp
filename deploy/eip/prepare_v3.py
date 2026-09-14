"""Build task processors and resolved V3 configs without initializing a model."""

import copy
import hashlib
import json
import os
from pathlib import Path

import draccus

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors


def main():
    """Keep the original data/model contract, with the reviewed V3 overrides."""
    repo = Path(__file__).resolve().parents[2]
    root = Path("/pfs/user/data/ctp_pi05_eip")
    initial = root / "models/pi05_base_pens168_v3_init"
    out = root / "configs/v3"
    out.mkdir(parents=True, exist_ok=True)
    initial.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((repo / "deploy/autodl/configs/stabilize.json").read_text())
    cfg["dataset"]["root"] = str(root / "data/pens168_v30")
    cfg["policy"].update(
        pretrained_path=str(initial),
        text_tokenizer_name=str(root / "models/paligemma_tokenizer"),
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
        checkpoint_keep_last=0,
        eval_steps=250,
        full_eval_steps=1000,
        eval_samples_per_episode=64,
        max_eval_samples=0,
        ctp_color_jitter=0.1,
        best_checkpoint_metric="ctp_selection_score",
        output_dir=str(root / "outputs/pi05_ctp_v3_20k_8xa800"),
        job_name="pi05_ctp_v3_20k_8xa800",
    )
    cfg["batch_size"] = 4
    cfg["parallelism"]["dp_shard"] = 8
    cfg["wandb"].update(
        run_id=None,
        resume=None,
        notes="V3 full finetuning from pinned official BASE: split CTP branches, detached width context, five learning rates, annealed responsibility temperature, exact masked validation. V3-only selection; checkpoint every2000.",
        enable=True,
        project="pi05-ctp-pen168",
        entity="shenchantian-harbin-institute-of-technology",
        disable_artifact=True,
        mode="online",
    )
    formal = draccus.decode(TrainPipelineConfig, cfg)
    formal.validate()
    formal.policy.save_pretrained(initial)
    meta = LeRobotDatasetMetadata(cfg["dataset"]["repo_id"], root=Path(cfg["dataset"]["root"]))
    pre, post = make_pi05_pre_post_processors(formal.policy, meta.stats)
    pre.save_pretrained(initial)
    post.save_pretrained(initial)
    base = root / "models/pi05_base/model.safetensors"
    link = initial / "model.safetensors"
    if not link.exists():
        os.link(base, link)
    if not link.samefile(base):
        raise RuntimeError("V3 initialization must link the verified official BASE")
    (out / "formal.json").write_text(json.dumps(formal.to_dict(), indent=2) + "\n")
    debug = copy.deepcopy(formal)
    debug.batch_size = 8
    debug.parallelism.dp_shard = 4
    debug.stop_after_step = 100
    debug.output_dir = root / "outputs/pi05_ctp_v3_debug_4xa800"
    debug.job_name = "pi05_ctp_v3_debug_4xa800"
    debug.validate()
    (out / "debug.json").write_text(json.dumps(debug.to_dict(), indent=2) + "\n")
    checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob("*.json")}
    print("V3_CONFIGS_READY", json.dumps(checksums), flush=True)


if __name__ == "__main__":
    main()
