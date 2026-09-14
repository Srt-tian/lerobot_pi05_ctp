"""Synthetic full-size CTP graph/optimizer probe; never a pretrained training run."""

import json
import os
import time
from pathlib import Path

os.environ["ACCELERATE_USE_FSDP"] = "true"
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin, set_seed

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def main():
    plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=["Linear", "Embedding"],
        reshard_after_forward=True,
    )
    accelerator = Accelerator(fsdp_plugin=plugin, mixed_precision="bf16")
    set_seed(42)
    device = accelerator.device
    x = torch.tensor([accelerator.process_index + 1.0], device=device)
    dist.all_reduce(x)
    assert x.item() == 10, x
    print("NCCL_OK", accelerator.process_index, flush=True)
    cfg = PI05Config(
        input_features={
            **{
                f"observation.images.cam_{c}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
                for c in ("front", "left", "right")
            },
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
        ctp_enabled=True,
        ctp_belief_cumulative_action_dims=0,
        dtype="float32",
        device=str(device),
        gradient_checkpointing=True,
        freeze_vision_encoder=False,
        train_expert_only=False,
        chunk_size=50,
    )
    policy = PI05Policy(cfg)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=2.5e-5, betas=(0.9, 0.95), eps=1e-8)
    print(
        "MODEL_INITIALIZED",
        accelerator.process_index,
        sum(p.numel() for p in policy.parameters()),
        flush=True,
    )
    policy, optimizer = accelerator.prepare(policy, optimizer)
    policy.train()
    batch = {key: torch.rand(1, 3, 224, 224, device=device) for key in cfg.image_features}
    batch[OBS_LANGUAGE_TOKENS] = torch.randint(1, 20000, (1, 200), device=device)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(1, 200, dtype=torch.bool, device=device)
    batch[ACTION] = torch.randn(1, 50, 14, device=device) * 0.05
    groups = ["vision_tower", "language_model", "gemma_expert", "ctp_parameter_head"]
    started = time.monotonic()
    with accelerator.autocast():
        loss, metrics = policy(batch)
    assert torch.isfinite(loss)
    accelerator.backward(loss)
    grad = torch.zeros(4, device=device)
    for name, param in policy.named_parameters():
        if param.grad is None:
            continue
        g = param.grad.to_local() if hasattr(param.grad, "to_local") else param.grad
        for i, group in enumerate(groups):
            if group in name:
                grad[i] += g.float().square().sum()
    dist.all_reduce(grad)
    assert torch.isfinite(grad).all() and (grad > 0).all(), grad
    accelerator.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    result = {
        "synthetic_only": True,
        "rank": accelerator.process_index,
        "loss": float(loss.detach()),
        "grad_norm_by_group": dict(zip(groups, grad.sqrt().tolist(), strict=True)),
        "seconds": time.monotonic() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }
    out = Path("/root/autodl-tmp/logs/full_graph_probe")
    out.mkdir(exist_ok=True)
    (out / f"rank{accelerator.process_index}.json").write_text(json.dumps(result, indent=2))
    print("FULL_GRAPH_STEP_OK", json.dumps(result), flush=True)
    accelerator.wait_for_everyone()
    accelerator.end_training()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
