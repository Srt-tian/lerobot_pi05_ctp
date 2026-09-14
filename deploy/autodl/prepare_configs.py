"""Create reproducible native LeRobot train configs and task-specific processors."""

import copy
import json
from pathlib import Path

from lerobot.configs import FeatureType
from lerobot.configs.accelerator import AcceleratorConfig, FSDPConfig
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.parallelism import ParallelismConfig
from lerobot.configs.train import CheckpointFormat, TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.utils.feature_utils import dataset_to_policy_features

root = Path("/root/autodl-tmp")
meta = LeRobotDatasetMetadata("local/pens168_ctp", root=root / "data/pens168_v30")
features = dataset_to_policy_features(meta.features)
init = root / "models/pi05_base_pens168_init"
init.mkdir(parents=True, exist_ok=True)
policy = PI05Config(
    input_features={k: v for k, v in features.items() if v.type != FeatureType.ACTION},
    output_features={k: v for k, v in features.items() if v.type == FeatureType.ACTION},
    pretrained_path=init,
    device="cuda",
    dtype="float32",
    push_to_hub=False,
    ctp_enabled=True,
    ctp_num_modes=4,
    ctp_belief_cumulative_action_dims=0,
    freeze_vision_encoder=False,
    train_expert_only=False,
    use_peft=False,
    use_relative_actions=False,
    gradient_checkpointing=True,
    compile_model=False,
    text_tokenizer_name=str(root / "models/paligemma_tokenizer"),
    chunk_size=50,
    n_action_steps=50,
    optimizer_lr=2.5e-5,
    scheduler_warmup_steps=1000,
    scheduler_decay_steps=20000,
)
policy.save_pretrained(init)
pre, post = make_pi05_pre_post_processors(policy, meta.stats)
pre.save_pretrained(init)
post.save_pretrained(init)
cfg = TrainPipelineConfig(
    dataset=DatasetConfig(
        repo_id="local/pens168_ctp", root=str(root / "data/pens168_v30"), eval_split=0.1, video_backend="pyav"
    ),
    policy=policy,
    output_dir=root / "outputs/pi05_ctp_pens168_full_20k",
    job_name="pi05_ctp_pens168_full_20k",
    batch_size=8,
    num_workers=4,
    prefetch_factor=2,
    persistent_workers=True,
    steps=20000,
    seed=42,
    tolerance_s=0.01,
    log_freq=10,
    env_eval_freq=0,
    eval_steps=1000,
    max_eval_samples=128,
    save_freq=1000,
    checkpoint_format=CheckpointFormat.DCP,
    checkpoint_keep_last=1,
    checkpoint_min_free_gb=10,
    parallelism=ParallelismConfig(dp_replicate=1, dp_shard=4),
    accelerator=AcceleratorConfig(
        mixed_precision="bf16", fsdp=FSDPConfig(wrap_modules=["Linear", "Embedding"])
    ),
    wandb=WandBConfig(
        enable=True, project="pi05-ctp-pen168", disable_artifact=True, mode="online", add_tags=False
    ),
)
out = Path(__file__).parent / "configs"
out.mkdir(exist_ok=True)
(out / "formal.json").write_text(json.dumps(cfg.to_dict(), indent=2))
debug = copy.deepcopy(cfg)
debug.output_dir = root / "outputs/pi05_ctp_pens168_debug"
debug.job_name = "pi05_ctp_pens168_debug"
debug.steps = 100
debug.save_freq = 50
debug.eval_steps = 50
debug.max_eval_samples = 32
(out / "debug.json").write_text(json.dumps(debug.to_dict(), indent=2))
print("CONFIGS_READY", str(out), flush=True)
