"""Regression tests for masked CTP likelihood and recoverable full finetuning."""

import copy
import json
import math
import shutil

import numpy as np
import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

from lerobot.common.ctp_training import parameter_groups, pin_best_model, stratified_indices
from lerobot.distributed.checkpoint import load_optimizer_with_lazy_states
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.pi05.ctp_loss import conditional_trajectory_peak_loss as loss_fn


def test_masked_likelihood_matches_torch_distribution_and_ignores_padding():
    """Test masked likelihood matches torch distribution and ignores padding."""
    torch.manual_seed(13)
    means = torch.randn(2, 4, 6, 3, dtype=torch.float64, requires_grad=True)
    target = torch.randn(2, 6, 3, dtype=torch.float64)
    logits = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    widths = torch.full((2, 4), 0.25, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[False] * 3 + [True] * 3, [False] * 5 + [True]])
    loss, _ = loss_fn(means, logits, widths, target, 0, action_is_pad=mask)
    for i, h in enumerate((3, 5)):
        dist = torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(logits=logits[i]),
            torch.distributions.Independent(
                torch.distributions.Normal(means[i, :, :h].reshape(4, -1), widths[i, :, None]), 1
            ),
        )
        expected = -dist.log_prob(target[i, :h].flatten()) / (h * 3) - 0.5 * math.log(2 * math.pi)
        torch.testing.assert_close(loss[i], expected)
    loss.sum().backward()
    assert means.grad[0, :, 3:].abs().sum() == 0
    assert means.grad[1, :, 5:].abs().sum() == 0
    assert means.grad[0, :, :3].abs().sum() > 0
    altered = target.clone()
    altered[mask] = 1e6
    a, _ = loss_fn(means, logits, widths, target, 0.01, action_is_pad=mask)
    b, _ = loss_fn(means, logits, widths, altered, 0.01, action_is_pad=mask)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError):
        loss_fn(means, logits, widths, target, 0, action_is_pad=torch.ones_like(mask))


def test_overlap_matches_cropped_trajectory_and_detach_removes_width_gradient():
    """Test overlap matches cropped trajectory and detach removes width gradient."""
    means = torch.randn(1, 4, 6, 3, dtype=torch.float64) * 0.05
    target = torch.randn(1, 6, 3, dtype=torch.float64)
    logits = torch.randn(1, 4, dtype=torch.float64, requires_grad=True)
    widths = torch.full((1, 4), 0.25, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[False] * 3 + [True] * 3])
    masked, metrics = loss_fn(
        means, logits, widths, target, 0.01, action_is_pad=mask, overlap_detach_widths=True
    )
    cropped, _ = loss_fn(means[:, :, :3], logits, widths, target[:, :3], 0.01, overlap_detach_widths=True)
    torch.testing.assert_close(masked, cropped)
    gradient = torch.autograd.grad(metrics["active_overlap"], widths, allow_unused=True)
    assert gradient[0] is None


class Tiny(torch.nn.Module):
    """Tiny."""

    def __init__(self):
        """Init  ."""
        super().__init__()
        self.model = torch.nn.Module()
        self.model.paligemma = torch.nn.Linear(2, 2)
        self.model.expert = torch.nn.Linear(2, 2)
        self.model.ctp_parameter_head = torch.nn.Linear(2, 2)

    def forward(self, x):
        """Forward."""
        return self.model.ctp_parameter_head(self.model.expert(self.model.paligemma(x))).square().mean()


def setup(model):
    """Setup."""
    groups = parameter_groups(model.named_parameters(), 1e-5, 0.25, 5.0)
    optimizer = torch.optim.AdamW(groups, lr=1e-5)
    schedule = CosineDecayWithWarmupSchedulerConfig(500, 8000, 1e-5, 1e-6).build(optimizer, 8000)
    return optimizer, schedule


def test_differential_lr_dcp_resume_matches_uninterrupted_update(tmp_path):
    """Test differential lr dcp resume matches uninterrupted update."""
    torch.manual_seed(13)
    model = Tiny()
    optimizer, scheduler = setup(model)
    x = torch.ones(2, 2)
    for _ in range(3):
        model(x).backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    state = get_optimizer_state_dict(model, optimizer)
    dcp.save({"optimizer": state}, checkpoint_id=tmp_path)
    resumed = copy.deepcopy(model)
    new_optimizer, new_scheduler = setup(resumed)
    load_optimizer_with_lazy_states(resumed, new_optimizer, tmp_path, StateDictOptions(), set())
    new_scheduler.load_state_dict(scheduler.state_dict())
    for network, opt, sched in ((model, optimizer, scheduler), (resumed, new_optimizer, new_scheduler)):
        network(x).backward()
        opt.step()
        sched.step()
    for a, b in zip(model.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert new_scheduler.get_last_lr() == scheduler.get_last_lr()
    lrs = dict(zip(("vlm", "expert", "head"), scheduler.get_last_lr(), strict=True))
    assert lrs["vlm"] / lrs["expert"] == 0.25
    assert lrs["head"] / lrs["expert"] == pytest.approx(5)
    assert scheduler.base_lrs == [2.5e-6, 1e-5, 5e-5]


def test_selection_matches_prior_fixed_protocol():
    """Test selection matches prior fixed protocol."""
    episodes = np.repeat(np.arange(151, 168), np.arange(100, 117))
    frames = np.concatenate([np.arange(n) for n in range(100, 117)])
    selected = stratified_indices(episodes, frames, 64)
    assert len(selected) == len(set(selected)) == 1088
    for e in range(151, 168):
        indices = np.flatnonzero(episodes == e)
        expected = indices[np.linspace(0, len(indices) - 1, 64, dtype=int)]
        assert [i for i in selected if episodes[i] == e] == expected.tolist()


def test_best_model_survives_latest_pruning_and_rejects_worse(tmp_path):
    """Test best model survives latest pruning and rejects worse."""

    def checkpoint(step):
        """Checkpoint."""
        p = tmp_path / f"{step:06}"
        (p / "pretrained_model").mkdir(parents=True)
        (p / "pretrained_model" / "weights").write_bytes(b"immutable-model")
        (p / "checkpoint_complete.json").write_text("{}")
        return p

    first = checkpoint(250)
    assert pin_best_model(first, {"mse": 0.1}, "mse")
    shutil.rmtree(first)
    assert (tmp_path / "best/pretrained_model/weights").read_bytes() == b"immutable-model"
    second = checkpoint(500)
    assert not pin_best_model(second, {"mse": 0.2}, "mse")
    assert pin_best_model(second, {"mse": 0.05}, "mse")
    selection = json.loads((tmp_path / "best/selection.json").read_text())
    assert selection["step"] == 500 and not selection["contains_optimizer"]
    assert not (tmp_path / "best.previous").exists()
