"""CTP joint-trajectory objective; source Wu-didi/ctp 5bdb960."""

import math

import torch
from torch import Tensor


def conditional_trajectory_peak_loss(
    trajectories: Tensor,
    logits: Tensor,
    widths: Tensor,
    target_actions: Tensor,
    overlap_weight: float,
    probability_floor: float = 0.0,
    entropy_target_effective_modes: float = 1.0,
    entropy_weight: float = 0.0,
    overlap_probability_weighted: bool = True,
    overlap_detach_widths: bool = False,
    action_is_pad: Tensor | None = None,
    temperature: float = 1.0,
    overlap_detach_probabilities: bool = False,
    return_per_sample_metrics: bool = False,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return per-sample CTP loss and diagnostics in joint trajectory space.

    Each Gaussian component owns one scalar width for the complete H x D
    action chunk.  The likelihood therefore assigns one component to the
    whole trajectory rather than independently switching components at each
    action step.  Dividing the joint NLL by the fixed trajectory dimension
    normalizes the likelihood scale. The overlap coefficient is therefore in
    per-coordinate NLL units; a paper joint-NLL coefficient must also be divided
    by H*D to preserve its relative strength.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("CTP temperature must be finite and positive")
    if trajectories.ndim != 4:
        raise ValueError("trajectories must have shape [B,K,H,D].")
    batch_size, num_modes, horizon, action_dim = trajectories.shape
    if target_actions.shape != (batch_size, horizon, action_dim):
        raise ValueError(
            "target_actions must match [B,H,D]: "
            f"expected {(batch_size, horizon, action_dim)}, got {tuple(target_actions.shape)}"
        )
    if logits.shape != (batch_size, num_modes) or widths.shape != (batch_size, num_modes):
        raise ValueError("CTP logits/widths do not match trajectory batch and mode dimensions.")

    errors = (trajectories - target_actions[:, None]).square()
    valid = torch.ones((batch_size, horizon), dtype=torch.bool, device=trajectories.device)
    if action_is_pad is not None:
        if action_is_pad.shape != valid.shape or action_is_pad.dtype != torch.bool:
            raise ValueError("action_is_pad must be bool [B,H].")
        valid = ~action_is_pad.to(trajectories.device)
        if not valid.any(dim=1).all():
            raise ValueError("Every trajectory must have at least one valid action.")
    dimensions = valid.sum(dim=1) * action_dim
    squared_error = (errors * valid[:, None, :, None]).sum(dim=(2, 3))
    safe_widths = widths.clamp_min(1.0e-12)
    component_nll = 0.5 * squared_error / safe_widths.square() + dimensions[:, None] * safe_widths.log()
    raw_probabilities = torch.softmax(logits, dim=-1)
    probabilities = (1.0 - float(probability_floor)) * raw_probabilities + (
        float(probability_floor) / float(num_modes)
    )
    log_joint = probabilities.clamp_min(1.0e-12).log() - component_nll
    per_sample_nll = -torch.logsumexp(log_joint, dim=-1) / dimensions

    per_sample_tempered_nll = -temperature * torch.logsumexp(log_joint / temperature, dim=-1) / dimensions
    responsibilities = torch.softmax(log_joint.detach() / temperature, dim=-1)
    pair_errors = (trajectories[:, :, None] - trajectories[:, None, :]).square()
    if action_is_pad is None:
        pair_mse = pair_errors.mean(dim=(3, 4))
    else:
        pair_mse = (pair_errors * valid[:, None, None, :, None]).sum(dim=(3, 4)) / dimensions[:, None, None]
    overlap_widths = widths.detach() if overlap_detach_widths else widths
    width_sum = overlap_widths.square()[:, :, None] + overlap_widths.square()[:, None, :]
    local_overlap = torch.exp(-pair_mse / (2.0 * width_sum.clamp_min(1.0e-12)))
    upper = torch.triu(
        torch.ones(num_modes, num_modes, dtype=torch.bool, device=logits.device),
        diagonal=1,
    )
    if num_modes > 1:
        if overlap_probability_weighted:
            overlap_probs = probabilities.detach() if overlap_detach_probabilities else probabilities
            pair_probability = overlap_probs[:, :, None] * overlap_probs[:, None, :]
            per_sample_overlap = (pair_probability * local_overlap)[:, upper].sum(dim=-1)
        else:
            # Every candidate pair must separate.  Mixture logits can no
            # longer evade this term by driving one component mass to zero.
            per_sample_overlap = local_overlap[:, upper].mean(dim=-1)
        pairwise_diversity = pair_mse[:, upper].mean()
    else:
        per_sample_overlap = per_sample_nll.new_zeros(batch_size)
        pairwise_diversity = per_sample_nll.new_zeros(())

    raw_probability_entropy = -(raw_probabilities * raw_probabilities.clamp_min(1.0e-12).log()).sum(dim=-1)
    target_entropy = math.log(float(entropy_target_effective_modes))
    per_sample_entropy_floor = (target_entropy - raw_probability_entropy).clamp_min(0.0).square()
    per_sample_loss = (
        per_sample_tempered_nll
        + float(overlap_weight) * per_sample_overlap
        + float(entropy_weight) * per_sample_entropy_floor
    )
    responsibility_usage = responsibilities.mean(dim=0)
    responsibility_entropy = -(responsibility_usage * responsibility_usage.clamp_min(1.0e-12).log()).sum()
    best_mse = squared_error.min(dim=-1).values / dimensions
    top = raw_probabilities.argmax(-1)
    batch_index = torch.arange(batch_size, device=trajectories.device)
    selected_error = errors[batch_index, top]
    selected_mse = squared_error[batch_index, top] / dimensions
    selected_width = widths[batch_index, top]
    prefix = min(10, horizon)
    prefix_count = valid[:, :prefix].sum(dim=1).clamp_min(1) * action_dim
    prefix_mse = (selected_error[:, :prefix] * valid[:, :prefix, None]).sum((1, 2)) / prefix_count
    diagnostics = {
        "mixture_nll": per_sample_nll.mean(),
        "tempered_nll": per_sample_tempered_nll.mean(),
        "responsibility_temperature": widths.new_tensor(temperature),
        "active_overlap": per_sample_overlap.mean(),
        "entropy_floor_penalty": per_sample_entropy_floor.mean(),
        "best_of_k_mse": best_mse.mean(),
        "top1_mse_frame_mean": selected_mse.mean(),
        "top1_mse": squared_error[batch_index, top].sum() / dimensions.sum(),
        "first_action_mse": selected_error[:, 0].mean(),
        "prefix10_mse_frame_mean": prefix_mse.mean(),
        "selected_peak_width": selected_width.mean(),
        "selected_standardized_mse": (selected_mse / selected_width.square()).mean(),
        "valid_fraction": valid.float().mean(),
        "mean_peak_width": widths.mean(),
        # Conditional effective K is computed per observation, rather than
        # from global head counts, so it cannot be inflated merely by using a
        # different single head in different states.
        "conditional_effective_k": raw_probability_entropy.exp().mean(),
        "batch_responsibility_effective_k": responsibility_entropy.exp(),
        "pairwise_trajectory_mse": pairwise_diversity,
        "max_mixture_probability": raw_probabilities.max(dim=-1).values.mean(),
        "effective_max_mixture_probability": probabilities.max(dim=-1).values.mean(),
    }
    for mode in range(num_modes):
        diagnostics[f"mode_{mode}_top1_fraction"] = (top == mode).float().mean()
        diagnostics[f"mode_{mode}_responsibility"] = responsibilities[:, mode].mean()
        diagnostics[f"mode_{mode}_probability"] = raw_probabilities[:, mode].mean()
    if return_per_sample_metrics:
        diagnostics = {
            "loss": per_sample_loss,
            "mixture_nll": per_sample_nll,
            "top1_mse": selected_mse,
            "prefix10_mse": prefix_mse,
            "first_action_mse": selected_error[:, 0].mean(-1),
            "best_of_k_mse": best_mse,
            "selected_peak_width": selected_width,
            "selected_standardized_mse": selected_mse / selected_width.square(),
            "conditional_effective_k": raw_probability_entropy.exp(),
            "mean_peak_width": widths.mean(-1),
            "valid_steps": valid.sum(-1),
            "prefix_steps": valid[:, :prefix].sum(-1),
        }
        # Individual action-coordinate errors allow gripper metrics without
        # assuming a specific robot's gripper indices inside the loss.
        for coordinate in range(action_dim):
            diagnostics[f"action_{coordinate}_mse"] = (selected_error[:, :, coordinate] * valid).sum(
                -1
            ) / valid.sum(-1)
        for mode in range(num_modes):
            diagnostics[f"mode_{mode}_top1_fraction"] = (top == mode).float()
            diagnostics[f"mode_{mode}_responsibility"] = responsibilities[:, mode]
            diagnostics[f"mode_{mode}_probability"] = raw_probabilities[:, mode]
    return per_sample_loss, diagnostics
