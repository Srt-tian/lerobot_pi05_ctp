"""Exact-coverage distributed CTP validation and reproducible train-only jitter."""

import math
from pathlib import Path

import torch
from torch.utils.data import Dataset

from lerobot.utils.random_utils import load_rng_state, save_rng_state


class PaddedValidationDataset(Dataset):
    """All ranks execute equally many forwards; artificial rows carry zero mass."""

    def __init__(self, dataset, global_batch_size):
        self.dataset = dataset
        self.real_size = len(dataset)
        if self.real_size <= 0:
            raise ValueError("Empty validation dataset")
        self.padded_size = math.ceil(self.real_size / global_batch_size) * global_batch_size

    def __len__(self):
        return self.padded_size

    def __getitem__(self, index):
        row = dict(self.dataset[index % self.real_size])
        row["ctp_eval_weight"] = float(index < self.real_size)
        return row


def color_jitter(batch, camera_keys, amount, seed):
    """Brightness/contrast/saturation only; seed is derived from step and rank.

    No worker RNG is involved, so prefetched batches across a process restart
    get exactly the same transform at the same step. Geometry is unchanged.
    """
    for index, key in enumerate(camera_keys):
        if key not in batch:
            continue
        image = batch[key]
        image = image.float() / 255 if image.dtype == torch.uint8 else image.float()
        generator = torch.Generator(device=image.device).manual_seed(seed + index)
        factors = (
            1
            + (torch.rand(3, image.shape[0], 1, 1, 1, generator=generator, device=image.device) * 2 - 1)
            * amount
        )
        image = (image * factors[0]).clamp(0, 1)
        weights = image.new_tensor([0.2989, 0.5870, 0.1140]).reshape(1, 3, 1, 1)
        gray = (image * weights).sum(1, keepdim=True)
        mean = gray.mean((-2, -1), keepdim=True)
        image = (mean + factors[1] * (image - mean)).clamp(0, 1)
        gray = (image * weights).sum(1, keepdim=True)
        batch[key] = (gray + factors[2] * (image - gray)).clamp(0, 1)
    return batch


def save_rank_rng(training_state_dir, accelerator):
    rank = accelerator.process_index if accelerator is not None else 0
    target = Path(training_state_dir) / f"rng_rank_{rank}"
    target.mkdir(parents=True, exist_ok=True)
    save_rng_state(target)


def restore_rank_rng(training_state_dir, accelerator):
    target = Path(training_state_dir) / f"rng_rank_{accelerator.process_index}"
    if not target.is_dir():
        raise FileNotFoundError(f"V3 resume requires per-rank RNG: {target}")
    load_rng_state(target)


def selection_score(metrics):
    """V3-only score in the common normalized action-coordinate space."""
    prefix, full = (float(metrics[k]) for k in ("ctp_prefix10_mse", "ctp_top1_mse"))
    if not all(math.isfinite(value) and value >= 0 for value in (prefix, full)):
        raise ValueError("Selection errors must be finite and nonnegative")
    return 0.7 * prefix + 0.3 * full


def accumulate_metrics(totals, metrics, weights, episodes, episode_ids):
    """Accumulate additive sufficient statistics, then reduce only once per pass."""
    for episode in [None, *episode_ids]:
        row_weights = weights.double() if episode is None else weights.double() * (episodes == episode)
        prefix = "" if episode is None else f"episode_{episode}/"
        for key, value in metrics.items():
            if key in ("valid_steps", "prefix_steps"):
                continue
            weight = row_weights
            if key in ("top1_mse", "mixture_nll") or key.startswith("action_"):
                weight = weight * metrics["valid_steps"]
            elif key == "prefix10_mse":
                weight = weight * metrics["prefix_steps"]
            name = prefix + "ctp_" + key
            contribution = torch.stack(((value.double() * weight).sum(), weight.sum()))
            totals[name] = totals.get(name, torch.zeros_like(contribution)) + contribution
        name = prefix + "samples"
        contribution = torch.stack((row_weights.sum(), row_weights.new_tensor(1)))
        totals[name] = totals.get(name, torch.zeros_like(contribution)) + contribution


def evaluate_ctp(policy, dataloader, accelerator, preprocess, episode_ids, expected_samples):
    policy.eval()
    totals = {}
    try:
        with torch.no_grad(), accelerator.autocast():
            for raw_batch in dataloader:
                weights = raw_batch.pop("ctp_eval_weight")
                episodes = raw_batch["episode_index"]
                batch = preprocess(raw_batch)
                _, metrics = policy(batch, reduction="none", return_per_sample_metrics=True)
                accumulate_metrics(
                    totals,
                    metrics,
                    weights.to(batch["action"].device),
                    episodes.to(batch["action"].device),
                    episode_ids,
                )
        keys = sorted(totals)
        packed = accelerator.reduce(torch.stack([totals[k] for k in keys]), reduction="sum").cpu().tolist()
        result = {
            key: total if key.endswith("samples") else total / count
            for key, (total, count) in zip(keys, packed, strict=True)
        }
        if result["samples"] != expected_samples:
            raise RuntimeError(f"Validation coverage mismatch: {result['samples']} != {expected_samples}")
        mode_keys = sorted(k for k in result if k.startswith("ctp_mode_") and k.endswith("_responsibility"))
        probs = torch.tensor([result[k] for k in mode_keys], dtype=torch.float64)
        result["ctp_batch_responsibility_effective_k"] = float(
            (-(probs * probs.clamp_min(1e-12).log()).sum()).exp()
        )
        return result
    finally:
        policy.train()
