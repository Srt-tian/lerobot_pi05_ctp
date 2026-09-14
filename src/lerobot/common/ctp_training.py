"""Bounded CTP training utilities with explicit checkpoint provenance."""

import json
import math
import os
import shutil
from pathlib import Path

import numpy as np


def parameter_groups(
    named_parameters, base_lr, vlm_scale, head_scale, *, logits_scale=None, width_scale=None
):
    split = logits_scale is not None and width_scale is not None
    groups = {
        name: []
        for name in (
            ("vlm", "expert", "ctp_centers", "ctp_logits", "ctp_widths")
            if split
            else ("vlm", "expert", "ctp_head")
        )
    }
    seen = set()
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError("Duplicate optimizer parameter")
        seen.add(id(parameter))
        group = (
            "ctp_head" if "ctp_parameter_head." in name else ("vlm" if ".paligemma." in name else "expert")
        )
        if split and group == "ctp_head":
            branch = name.split("ctp_parameter_head.", 1)[1].split(".", 1)[0]
            group = "ctp_" + branch
            if group not in groups:
                raise ValueError(f"Unknown split CTP parameter: {name}")
        groups[group].append(parameter)
    scales = {
        "vlm": vlm_scale,
        "expert": 1.0,
        "ctp_head": head_scale,
        "ctp_centers": head_scale,
        "ctp_logits": logits_scale,
        "ctp_widths": width_scale,
    }
    if not all(groups.values()):
        raise ValueError("Differential full-finetuning requires VLM, expert and head parameters")
    return [
        {"params": values, "lr": base_lr * scales[name], "group_name": name}
        for name, values in groups.items()
    ]


def stratified_indices(episode_ids, frame_ids, samples_per_episode):
    """Equally spaced frames including both episode endpoints; stable order."""
    selected = []
    for episode in sorted(set(episode_ids.tolist())):
        indices = [i for i, e in enumerate(episode_ids) if e == episode]
        indices.sort(key=lambda i: frame_ids[i])
        count = min(samples_per_episode, len(indices))
        positions = np.linspace(0, len(indices) - 1, count, dtype=int).tolist()
        selected.extend(indices[position] for position in positions)
    return selected


def pin_best_model(checkpoint, metrics, metric_name):
    """Retain only model shards, via hardlinks, alongside latest resumable state."""
    checkpoint = Path(checkpoint)
    metric = float(metrics[metric_name])
    if not math.isfinite(metric):
        raise ValueError("Non-finite checkpoint selection metric")
    best = checkpoint.parent / "best"
    previous = checkpoint.parent / "best.previous"
    temporary = checkpoint.parent / "best.partial"
    if best.exists():
        old = json.loads((best / "selection.json").read_text())
        if metric >= old["value"]:
            return False
    if temporary.exists() or previous.exists():
        raise RuntimeError("Unfinished best-model transaction requires inspection")
    if not (checkpoint / "checkpoint_complete.json").is_file():
        raise RuntimeError("Only completed checkpoints can be pinned")
    temporary.mkdir()
    shutil.copytree(checkpoint / "pretrained_model", temporary / "pretrained_model", copy_function=os.link)
    (temporary / "selection.json").write_text(
        json.dumps(
            {
                "step": int(checkpoint.name),
                "metric": metric_name,
                "value": metric,
                "metrics": metrics,
                "source": str(checkpoint),
                "contains_optimizer": False,
            },
            indent=2,
        )
        + "\n"
    )
    if best.exists():
        best.rename(previous)
    temporary.rename(best)
    if previous.exists():
        shutil.rmtree(previous)
    return True
