"""Independent checks for held-out coverage and metric aggregation."""

import numpy as np
from eval_metrics import frame_metrics, select_frames, summarize


def test_selection_covers_every_episode_and_both_ends():
    """Verify selection covers every episode and both ends."""
    episodes = np.repeat([151, 152, 153], [5, 7, 3])
    frames = np.concatenate([np.arange(5), np.arange(7), np.arange(3)])
    assert select_frames(episodes, frames) == list(range(15))
    selected = select_frames(episodes, frames, 2)
    assert selected == [0, 4, 5, 11, 12, 14]


def test_top1_is_not_oracle_and_padding_is_excluded():
    """Verify top1 is not oracle and padding is excluded."""
    target = np.zeros((2, 3, 2))
    candidates = np.zeros((2, 2, 3, 2))
    candidates[:, 0] = 2
    candidates[0, :, 1:] = 999
    probabilities = np.array([[0.9, 0.1], [0.9, 0.1]])
    valid = np.array([[True, False, False], [True, True, True]])
    rows = frame_metrics(
        candidates, probabilities, np.ones((2, 2)), target, candidates[:, 0], target, np.zeros((2, 2)), valid
    )
    assert rows[0]["top1_normalized_mse"] == 4
    assert rows[0]["best_of_k_normalized_mse"] == 0
    report = summarize(rows)
    assert report["top1_normalized_mse"] == 4
    assert report["raw_joint_rmse"] == [2, 2]
    assert report["valid_action_steps"] == 4
    assert report["top1_mode_counts"] == [2, 0]


def test_valid_action_weighted_aggregation():
    """Verify valid action weighted aggregation."""
    target = np.zeros((2, 3, 1))
    predictions = np.ones((2, 1, 3, 1))
    predictions[0] = 2
    valid = np.array([[True, False, False], [True, True, True]])
    rows = frame_metrics(
        predictions,
        np.ones((2, 1)),
        np.ones((2, 1)),
        target,
        predictions[:, 0],
        target,
        np.zeros((2, 1)),
        valid,
    )
    assert summarize(rows)["top1_normalized_mse"] == 1.75
