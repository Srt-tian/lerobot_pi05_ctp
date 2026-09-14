"""Offline CTP metrics: valid horizon masking and explicit top-1 semantics."""

import numpy as np


def select_frames(episodes, frames, per_episode=0):
    """Zero means every frame; otherwise evenly cover each complete episode."""
    episodes, frames = np.asarray(episodes), np.asarray(frames)
    selected = []
    for episode in np.unique(episodes):
        indices = np.flatnonzero(episodes == episode)
        indices = indices[np.argsort(frames[indices])]
        if per_episode > 0 and len(indices) > per_episode:
            indices = indices[np.linspace(0, len(indices) - 1, per_episode, dtype=int)]
        selected.extend(indices.tolist())
    return selected


def frame_metrics(trajectories, probabilities, widths, target, raw_prediction, raw_target, raw_state, valid):
    """One row per observation; raw joint units remain separate (no mixed-unit score)."""
    b, k, h, d = trajectories.shape
    assert valid.shape == (b, h) and valid[:, 0].all()
    assert np.isfinite(trajectories).all() and np.isfinite(raw_prediction).all()
    top = probabilities.argmax(-1)
    predicted = trajectories[np.arange(b), top]
    mask = valid[..., None]
    count = valid.sum(-1) * d
    squared = ((trajectories - target[:, None]) ** 2 * mask[:, None]).sum(axis=(2, 3))
    raw_error = raw_prediction - raw_target
    baseline_error = raw_state[:, None, :d] - raw_target
    nll = 0.5 * squared / widths**2 + count[:, None] * np.log(widths)
    log_joint = np.log(np.maximum(probabilities, 1e-12)) - nll
    shifted = log_joint - log_joint.max(-1, keepdims=True)
    responsibility = np.exp(shifted) / np.exp(shifted).sum(-1, keepdims=True)
    masked_nll = -(log_joint.max(-1) + np.log(np.exp(shifted).sum(-1))) / count
    result = []
    for i in range(b):
        horizon_mse = np.mean((predicted[i] - target[i]) ** 2, axis=-1)
        result.append(
            {
                "valid_horizon_steps": int(valid[i].sum()),
                "top1_mode": int(top[i]),
                "responsibility_mode": int(responsibility[i].argmax()),
                "probabilities": probabilities[i].tolist(),
                "responsibilities": responsibility[i].tolist(),
                "conditional_effective_k": float(
                    np.exp(-(probabilities[i] * np.log(np.maximum(probabilities[i], 1e-12))).sum())
                ),
                "masked_mixture_nll": float(masked_nll[i]),
                "top1_normalized_mse": float(squared[i, top[i]] / count[i]),
                "best_of_k_normalized_mse": float(squared[i].min() / count[i]),
                "first_action_normalized_mse": float(horizon_mse[0]),
                "horizon_normalized_mse": {
                    str(t): float(horizon_mse[t]) for t in [0, 9, 24, 49] if t < h and valid[i, t]
                },
                "raw_joint_abs_sum": (np.abs(raw_error[i]) * mask[i]).sum(0).tolist(),
                "raw_joint_squared_sum": (raw_error[i] ** 2 * mask[i]).sum(0).tolist(),
                "hold_state_joint_squared_sum": (baseline_error[i] ** 2 * mask[i]).sum(0).tolist(),
            }
        )
    return result


def summarize(rows):
    """Aggregate by valid action count and preserve per-joint units."""
    counts = np.asarray([r["valid_horizon_steps"] for r in rows])
    modes = np.asarray([r["top1_mode"] for r in rows])
    k = len(rows[0]["probabilities"])
    probabilities = np.asarray([r["probabilities"] for r in rows]).mean(0)
    responsibilities = np.asarray([r["responsibilities"] for r in rows]).mean(0)
    result = {
        "frames": len(rows),
        "valid_action_steps": int(counts.sum()),
        "top1_mode_counts": np.bincount(modes, minlength=k).tolist(),
        "mean_probabilities": probabilities.tolist(),
        "mean_responsibilities": responsibilities.tolist(),
        "responsibility_effective_k": float(
            np.exp(-(responsibilities * np.log(np.maximum(responsibilities, 1e-12))).sum())
        ),
    }
    for key in ["top1_normalized_mse", "best_of_k_normalized_mse", "masked_mixture_nll"]:
        result[key] = float(np.average([r[key] for r in rows], weights=counts))
    for key in ["first_action_normalized_mse", "conditional_effective_k", "legacy_loss"]:
        if key in rows[0]:
            result[key] = float(np.mean([r[key] for r in rows]))
    for output, key, square_root in [
        ("raw_joint_mae", "raw_joint_abs_sum", False),
        ("raw_joint_rmse", "raw_joint_squared_sum", True),
        ("hold_state_joint_rmse", "hold_state_joint_squared_sum", True),
    ]:
        value = np.asarray([r[key] for r in rows]).sum(0) / counts.sum()
        result[output] = (np.sqrt(value) if square_root else value).tolist()
    for h in ["0", "9", "24", "49"]:
        values = [r["horizon_normalized_mse"][h] for r in rows if h in r["horizon_normalized_mse"]]
        if values:
            result[f"horizon_{h}_normalized_mse"] = float(np.mean(values))
    result["top1_frame_rmse_p50_p90_p95"] = np.quantile(
        np.sqrt([r["top1_normalized_mse"] for r in rows]), [0.5, 0.9, 0.95]
    ).tolist()
    return result
