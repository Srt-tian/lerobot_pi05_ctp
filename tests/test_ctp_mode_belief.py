from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).parents[1] / "src/lerobot/policies/pi05/ctp_mode_belief.py"
SPEC = importlib.util.spec_from_file_location("ctp_mode_belief", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

TrajectoryMatchedModeBelief = MODULE.TrajectoryMatchedModeBelief
trajectory_overlap_cost = MODULE.trajectory_overlap_cost


def _candidate(delta: float) -> torch.Tensor:
    return torch.full((4, 1), delta, dtype=torch.float32)


def _selector(**overrides) -> TrajectoryMatchedModeBelief:
    kwargs = {
        "inference_mode": "path_belief",
        "execute_horizon": 2,
        "cumulative_action_dims": 1,
        "self_transition": 1.0,
        "observation_weight": 1.0,
        "release_ratio": 4.0,
        "transport_temperature": 0.05,
        "confidence_threshold": 0.0,
        "probability_floor": 0.0,
    }
    kwargs.update(overrides)
    return TrajectoryMatchedModeBelief(**kwargs)


def test_relative_path_cost_matches_continuation() -> None:
    positive = _candidate(1.0)
    negative = _candidate(-1.0)
    previous = torch.stack([positive, negative])[None]
    current = torch.stack([negative, positive])[None]

    cost = trajectory_overlap_cost(
        previous,
        current,
        execute_horizon=2,
        cumulative_action_dims=1,
    )

    assert cost[0, 0, 1] < cost[0, 0, 0]
    assert cost[0, 1, 0] < cost[0, 1, 1]


def test_component_permutation_is_not_a_behavior_switch() -> None:
    positive = _candidate(1.0)
    negative = _candidate(-1.0)
    selector = _selector()

    first = selector.select(
        torch.tensor([[8.0, 0.0]]),
        torch.stack([positive, negative])[None],
    )
    second = selector.select(
        torch.tensor([[0.0, 0.0]]),
        torch.stack([negative, positive])[None],
    )

    assert first.selected.item() == 0
    assert second.selected.item() == 1
    assert second.raw_switch.item()
    assert not second.path_switch.item()
    assert second.matched_continuation.item() == 1


def test_release_allows_necessary_replanning() -> None:
    positive = _candidate(1.0)
    negative = _candidate(-1.0)
    candidates = torch.stack([positive, negative])[None]
    selector = _selector()

    selector.select(torch.tensor([[8.0, 0.0]]), candidates)
    step = selector.select(torch.tensor([[0.0, 8.0]]), candidates)

    assert step.released.item()
    assert step.selected.item() == 1
    assert step.path_switch.item()


def test_low_confidence_transport_falls_back_to_current_evidence() -> None:
    duplicate = _candidate(1.0)
    candidates = torch.stack([duplicate, duplicate])[None]
    selector = _selector(confidence_threshold=0.9)

    selector.select(torch.tensor([[8.0, 0.0]]), candidates)
    step = selector.select(torch.tensor([[0.0, 8.0]]), candidates)

    assert step.confidence_fallback.item()
    assert step.selected.item() == 1


if __name__ == "__main__":
    tests = [
        test_relative_path_cost_matches_continuation,
        test_component_permutation_is_not_a_behavior_switch,
        test_release_allows_necessary_replanning,
        test_low_confidence_transport_falls_back_to_current_evidence,
    ]
    for test in tests:
        test()
    print(f"passed {len(tests)} CTP mode-belief tests")
