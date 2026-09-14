"""Trajectory-matched temporal decoding for PI0.5 Conditional Trajectory Peaks.

The CTP components are exchangeable: component ``k`` at one replanning step
does not have to denote the same behavior as component ``k`` at the next.  The
decoder below therefore transports belief through trajectory geometry instead
of component indices.  It is inference-only and does not modify policy
weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


def trajectory_overlap_cost(
    previous: Tensor,
    current: Tensor,
    *,
    execute_horizon: int,
    cumulative_action_dims: int,
) -> Tensor:
    """Return pairwise continuation cost for two exchangeable candidate sets.

    Args:
        previous: Previous candidates with shape ``[B, K, H, D]``.
        current: Current candidates with the same shape.
        execute_horizon: Number of previous actions already executed.
        cumulative_action_dims: Leading relative-control dimensions to
            integrate before matching.  LIBERO uses six relative end-effector
            pose controls followed by one non-integrated gripper command.

    The previous unexecuted suffix and current time-aligned prefix are compared
    in a common, re-anchored path space.  Re-anchoring is essential for
    relative actions: the current plan starts at the new observation, whereas
    the previous plan started ``execute_horizon`` steps earlier.
    """

    if previous.ndim != 4 or current.ndim != 4:
        raise ValueError("CTP candidates must have shape [B,K,H,D].")
    if previous.shape != current.shape:
        raise ValueError("Previous and current candidate sets must have identical shape.")
    _, _, horizon, action_dim = previous.shape
    execute_horizon = int(execute_horizon)
    cumulative_action_dims = int(cumulative_action_dims)
    if not 1 <= execute_horizon < horizon:
        raise ValueError("execute_horizon must be in [1, H).")
    if not 0 <= cumulative_action_dims <= action_dim:
        raise ValueError("cumulative_action_dims must be in [0, D].")
    if not torch.isfinite(previous).all() or not torch.isfinite(current).all():
        raise ValueError("CTP candidates must be finite.")

    overlap = horizon - execute_horizon
    pieces_previous: list[Tensor] = []
    pieces_current: list[Tensor] = []

    if cumulative_action_dims:
        previous_path = previous[..., :cumulative_action_dims].cumsum(dim=2)
        current_path = current[..., :cumulative_action_dims].cumsum(dim=2)
        previous_origin = previous_path[:, :, execute_horizon - 1 : execute_horizon]
        pieces_previous.append(previous_path[:, :, execute_horizon:] - previous_origin)
        pieces_current.append(current_path[:, :, :overlap])

    if cumulative_action_dims < action_dim:
        pieces_previous.append(previous[:, :, execute_horizon:, cumulative_action_dims:])
        pieces_current.append(current[:, :, :overlap, cumulative_action_dims:])

    previous_suffix = torch.cat(pieces_previous, dim=-1)
    current_prefix = torch.cat(pieces_current, dim=-1)
    return (previous_suffix[:, :, None] - current_prefix[:, None, :]).square().mean(dim=(3, 4))


def row_conditional_correspondence(pairwise_cost: Tensor, *, temperature: float) -> Tensor:
    """Map every previous candidate to current candidates without column balancing.

    A row-conditional soft correspondence is permutation equivariant but does
    not force unrelated previous paths to occupy distinct current columns.
    This avoids the failure mode of balanced Sinkhorn matching observed in the
    earlier D3IL ablation.
    """

    if pairwise_cost.ndim != 3 or pairwise_cost.shape[1] != pairwise_cost.shape[2]:
        raise ValueError("pairwise_cost must have shape [B,K,K].")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive.")
    eps = torch.finfo(pairwise_cost.dtype).eps
    scales: list[Tensor] = []
    for cost in pairwise_cost.detach():
        positive = cost[cost > eps]
        scales.append(positive.median().clamp_min(eps) if positive.numel() else cost.new_tensor(1.0))
    scale = torch.stack(scales).view(pairwise_cost.shape[0], 1, 1)
    return torch.softmax(-pairwise_cost / scale / float(temperature), dim=-1)


@dataclass
class ModeBeliefStep:
    selected: Tensor
    probabilities: Tensor
    posterior: Tensor
    matched_continuation: Tensor
    raw_switch: Tensor
    path_switch: Tensor
    released: Tensor
    confidence_fallback: Tensor
    transport_confidence: Tensor
    matched_cost: Tensor


class TrajectoryMatchedModeBelief:
    """Episode-stateful, training-free selector for exchangeable CTP candidates."""

    def __init__(
        self,
        *,
        inference_mode: str,
        execute_horizon: int,
        cumulative_action_dims: int,
        self_transition: float,
        observation_weight: float,
        release_ratio: float,
        transport_temperature: float,
        confidence_threshold: float,
        probability_floor: float,
    ) -> None:
        if inference_mode not in {"top1", "path_belief"}:
            raise ValueError("inference_mode must be 'top1' or 'path_belief'.")
        if not 0.0 <= float(self_transition) <= 1.0:
            raise ValueError("self_transition must be in [0, 1].")
        if not math.isfinite(float(observation_weight)) or float(observation_weight) <= 0.0:
            raise ValueError("observation_weight must be finite and positive.")
        if float(release_ratio) != 0.0 and (
            not math.isfinite(float(release_ratio)) or float(release_ratio) < 1.0
        ):
            raise ValueError("release_ratio must be 0 or at least 1.")
        if not 0.0 <= float(confidence_threshold) <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1].")
        if not 0.0 <= float(probability_floor) < 1.0:
            raise ValueError("probability_floor must be in [0, 1).")

        self.inference_mode = inference_mode
        self.execute_horizon = int(execute_horizon)
        self.cumulative_action_dims = int(cumulative_action_dims)
        self.self_transition = float(self_transition)
        self.observation_weight = float(observation_weight)
        self.release_ratio = float(release_ratio)
        self.transport_temperature = float(transport_temperature)
        self.confidence_threshold = float(confidence_threshold)
        self.probability_floor = float(probability_floor)

        self._previous_candidates: Tensor | None = None
        self._previous_selected: Tensor | None = None
        self._previous_belief: Tensor | None = None
        self._totals = {
            "decisions": 0,
            "transport_decisions": 0,
            "raw_switches": 0,
            "path_switches": 0,
            "releases": 0,
            "confidence_fallbacks": 0,
            "transport_confidence_sum": 0.0,
            "matched_cost_sum": 0.0,
        }

    def reset_episode(self) -> None:
        self._previous_candidates = None
        self._previous_selected = None
        self._previous_belief = None

    def summary(self) -> dict[str, float | int | str]:
        transports = int(self._totals["transport_decisions"])
        return {
            "inference_mode": self.inference_mode,
            **self._totals,
            "mean_transport_confidence": float(self._totals["transport_confidence_sum"]) / max(transports, 1),
            "mean_matched_cost": float(self._totals["matched_cost_sum"]) / max(transports, 1),
        }

    def select(self, logits: Tensor, candidates: Tensor) -> ModeBeliefStep:
        if logits.ndim != 2 or candidates.ndim != 4:
            raise ValueError("Expected logits [B,K] and candidates [B,K,H,D].")
        batch, modes = logits.shape
        if candidates.shape[:2] != (batch, modes):
            raise ValueError("Logits and candidate set dimensions do not match.")
        if modes < 1:
            raise ValueError("At least one CTP mode is required.")

        raw_probabilities = torch.softmax(logits, dim=-1)
        probabilities = (1.0 - self.probability_floor) * raw_probabilities + (
            self.probability_floor / float(modes)
        )
        likelihood = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).pow(
            self.observation_weight
        )
        posterior = likelihood
        device = logits.device
        batch_index = torch.arange(batch, device=device)
        invalid_index = torch.full((batch,), -1, dtype=torch.long, device=device)
        matched_continuation = invalid_index.clone()
        raw_switch = torch.zeros(batch, dtype=torch.bool, device=device)
        path_switch = torch.zeros_like(raw_switch)
        released = torch.zeros_like(raw_switch)
        confidence_fallback = torch.zeros_like(raw_switch)
        confidence = torch.zeros(batch, dtype=logits.dtype, device=device)
        matched_cost = torch.zeros_like(confidence)

        previous_candidates = self._previous_candidates
        previous_selected = self._previous_selected
        previous_belief = self._previous_belief
        transport_valid = (
            previous_candidates is not None
            and previous_selected is not None
            and previous_belief is not None
            and previous_candidates.shape == candidates.shape
            and previous_belief.shape == probabilities.shape
        )
        if transport_valid:
            pairwise_cost = trajectory_overlap_cost(
                previous_candidates,
                candidates,
                execute_horizon=self.execute_horizon,
                cumulative_action_dims=self.cumulative_action_dims,
            )
            transition = row_conditional_correspondence(pairwise_cost, temperature=self.transport_temperature)
            selected_row = transition[batch_index, previous_selected]
            matched_continuation = selected_row.argmax(dim=-1)
            confidence = selected_row.max(dim=-1).values
            matched_cost = pairwise_cost[batch_index, previous_selected, matched_continuation]

            if self.inference_mode == "path_belief" and modes > 1:
                transported_belief = torch.einsum("bi,bij->bj", previous_belief, transition)
                prior = self.self_transition * transported_belief + (1.0 - self.self_transition) / float(
                    modes
                )
                posterior = prior * likelihood

                challenger = likelihood.argmax(dim=-1)
                if self.release_ratio > 0.0:
                    evidence_ratio = likelihood[batch_index, challenger] / likelihood[
                        batch_index, matched_continuation
                    ].clamp_min(torch.finfo(probabilities.dtype).tiny)
                    released = challenger.ne(matched_continuation) & evidence_ratio.ge(self.release_ratio)
                    posterior = torch.where(released[:, None], likelihood, posterior)

                confidence_fallback = confidence.lt(self.confidence_threshold)
                released = released & ~confidence_fallback
                posterior = torch.where(confidence_fallback[:, None], likelihood, posterior)

        posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(probabilities.dtype).tiny
        )
        selected = posterior.argmax(dim=-1)
        if transport_valid:
            raw_switch = selected.ne(previous_selected)
            path_switch = selected.ne(matched_continuation)

        self._totals["decisions"] += batch
        if transport_valid:
            self._totals["transport_decisions"] += batch
            self._totals["raw_switches"] += int(raw_switch.sum().item())
            self._totals["path_switches"] += int(path_switch.sum().item())
            self._totals["releases"] += int(released.sum().item())
            self._totals["confidence_fallbacks"] += int(confidence_fallback.sum().item())
            self._totals["transport_confidence_sum"] += float(confidence.sum().item())
            self._totals["matched_cost_sum"] += float(matched_cost.sum().item())

        self._previous_candidates = candidates.detach().clone()
        self._previous_selected = selected.detach().clone()
        self._previous_belief = posterior.detach().clone()
        return ModeBeliefStep(
            selected=selected,
            probabilities=probabilities,
            posterior=posterior,
            matched_continuation=matched_continuation,
            raw_switch=raw_switch,
            path_switch=path_switch,
            released=released,
            confidence_fallback=confidence_fallback,
            transport_confidence=confidence,
            matched_cost=matched_cost,
        )
