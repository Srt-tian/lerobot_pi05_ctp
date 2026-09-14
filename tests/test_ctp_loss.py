from __future__ import annotations

import math

import torch

from lerobot.policies.pi05.ctp_loss import conditional_trajectory_peak_loss


def _old_ctp_loss(
    trajectories: torch.Tensor,
    logits: torch.Tensor,
    widths: torch.Tensor,
    target: torch.Tensor,
    overlap_weight: float,
) -> torch.Tensor:
    _, modes, horizon, action_dim = trajectories.shape
    trajectory_dim = horizon * action_dim
    squared_error = (trajectories - target[:, None]).square().sum(dim=(2, 3))
    component_nll = 0.5 * squared_error / widths.square() + trajectory_dim * widths.log()
    nll = -torch.logsumexp(torch.log_softmax(logits, dim=-1) - component_nll, dim=-1)
    nll = nll / float(trajectory_dim)
    pair_mse = (trajectories[:, :, None] - trajectories[:, None, :]).square().mean(dim=(3, 4))
    width_sum = widths.square()[:, :, None] + widths.square()[:, None, :]
    overlap = torch.exp(-pair_mse / (2.0 * width_sum))
    probabilities = torch.softmax(logits, dim=-1)
    pair_probability = probabilities[:, :, None] * probabilities[:, None, :]
    upper = torch.triu(
        torch.ones(modes, modes, dtype=torch.bool),
        diagonal=1,
    )
    return nll + overlap_weight * (pair_probability * overlap)[:, upper].sum(dim=-1)


def test_legacy_defaults_are_numerically_equivalent() -> None:
    torch.manual_seed(7)
    trajectories = torch.randn(2, 4, 3, 2)
    logits = torch.randn(2, 4)
    widths = torch.rand(2, 4) * 0.3 + 0.1
    target = torch.randn(2, 3, 2)

    expected = _old_ctp_loss(trajectories, logits, widths, target, 0.1)
    actual, _ = conditional_trajectory_peak_loss(trajectories, logits, widths, target, overlap_weight=0.1)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_unweighted_detached_overlap_cannot_escape_through_mass_or_width() -> None:
    torch.manual_seed(9)
    trajectories = torch.randn(2, 4, 3, 2, requires_grad=True)
    logits = torch.randn(2, 4, requires_grad=True)
    widths = torch.full((2, 4), 0.2, requires_grad=True)
    target = torch.randn(2, 3, 2)

    _, diagnostics = conditional_trajectory_peak_loss(
        trajectories,
        logits,
        widths,
        target,
        overlap_weight=1.0,
        overlap_probability_weighted=False,
        overlap_detach_widths=True,
    )
    diagnostics["active_overlap"].backward()

    assert trajectories.grad is not None and trajectories.grad.abs().sum() > 0
    assert logits.grad is None or torch.count_nonzero(logits.grad) == 0
    assert widths.grad is None or torch.count_nonzero(widths.grad) == 0


def test_entropy_floor_penalizes_collapsed_conditional_mixture() -> None:
    trajectories = torch.zeros(1, 4, 3, 2)
    widths = torch.full((1, 4), 0.2)
    target = torch.zeros(1, 3, 2)
    collapsed = torch.tensor([[12.0, -12.0, -12.0, -12.0]])
    balanced = torch.zeros(1, 4)

    _, collapsed_metrics = conditional_trajectory_peak_loss(
        trajectories,
        collapsed,
        widths,
        target,
        overlap_weight=0.0,
        entropy_target_effective_modes=2.0,
        entropy_weight=1.0,
    )
    _, balanced_metrics = conditional_trajectory_peak_loss(
        trajectories,
        balanced,
        widths,
        target,
        overlap_weight=0.0,
        entropy_target_effective_modes=2.0,
        entropy_weight=1.0,
    )

    assert collapsed_metrics["entropy_floor_penalty"] > 0.9 * math.log(2.0) ** 2
    assert balanced_metrics["entropy_floor_penalty"] == 0


if __name__ == "__main__":
    tests = [
        test_legacy_defaults_are_numerically_equivalent,
        test_unweighted_detached_overlap_cannot_escape_through_mass_or_width,
        test_entropy_floor_penalizes_collapsed_conditional_mixture,
    ]
    for test in tests:
        test()
    print(f"passed {len(tests)} CTP loss tests")
