"""V3 mathematical, coverage and native optimizer recovery acceptance tests."""

import copy
import json

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

from lerobot.common.checkpoint_retention import MARKER, mark_complete_and_prune
from lerobot.common.ctp_training import parameter_groups
from lerobot.common.ctp_validation import PaddedValidationDataset, accumulate_metrics, color_jitter
from lerobot.distributed.checkpoint import load_optimizer_with_lazy_states
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.pi05.ctp_head import SplitCTPHead, responsibility_temperature
from lerobot.policies.pi05.ctp_loss import conditional_trajectory_peak_loss as loss_fn


def head():
    return SplitCTPHead(8, 4, 21, 0.0002, 0.1, 0.25, 0.5, True)


def test_split_head_preserves_capacity_and_stops_only_width_context_gradient():
    model = head().double()
    assert sum(p.numel() for p in model.parameters()) == (8 + 1) * 4 * (21 + 2)
    context = torch.randn(2, 8, dtype=torch.double, requires_grad=True)
    raw = model(context).reshape(2, 4, 23)
    gradient = torch.autograd.grad(
        raw[..., -1].sum(), (context, model.widths.weight), allow_unused=True, retain_graph=True
    )
    assert (gradient[0] is None or gradient[0].abs().sum() == 0) and gradient[1].abs().sum() > 0
    grad = torch.autograd.grad(raw[..., :-1].sum(), context)[0]
    assert grad.abs().sum() > 0
    fused = torch.nn.Linear(8, 4 * 23).double()
    with torch.no_grad():
        fused.weight.copy_(
            torch.cat(
                (
                    model.centers.weight.reshape(4, 21, 8),
                    model.logits.weight[:, None],
                    model.widths.weight[:, None],
                ),
                1,
            ).flatten(0, 1)
        )
        fused.bias.copy_(
            torch.cat(
                (model.centers.bias.reshape(4, 21), model.logits.bias[:, None], model.widths.bias[:, None]), 1
            ).flatten()
        )
    torch.testing.assert_close(model(context), fused(context))


@pytest.mark.parametrize("temperature", [1.0, 4.0, 20.0])
def test_tempered_likelihood_matches_independent_log_mixture(temperature):
    torch.manual_seed(15)
    means = torch.randn(2, 4, 7, 3, dtype=torch.double, requires_grad=True)
    widths = torch.rand(2, 4, dtype=torch.double) * 0.2 + 0.1
    logits = torch.randn(2, 4, dtype=torch.double)
    target = torch.randn(2, 7, 3, dtype=torch.double)
    mask = torch.tensor([[False] * 3 + [True] * 4, [False] * 7])
    loss, _ = loss_fn(means, logits, widths, target, 0, temperature=temperature, action_is_pad=mask)
    for row, length in enumerate((3, 7)):
        component = torch.distributions.Independent(
            torch.distributions.Normal(means[row, :, :length], widths[row, :, None, None]), 2
        )
        log_joint = component.log_prob(target[row, :length]) + logits[row].log_softmax(-1)
        expected = -temperature * (log_joint / temperature).logsumexp(-1) / (length * 3)
        expected -= 0.5 * torch.log(torch.tensor(2 * torch.pi, dtype=torch.double))
        torch.testing.assert_close(loss[row], expected)
    loss.sum().backward()
    assert means.grad[0, :, 3:].abs().sum() == 0


def test_overlap_gradient_only_moves_centers():
    means = (torch.randn(2, 4, 7, 3, dtype=torch.double) * 0.05).requires_grad_()
    logits = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
    widths = torch.full((2, 4), 0.25, dtype=torch.double, requires_grad=True)
    _, metrics = loss_fn(
        means,
        logits,
        widths,
        torch.zeros(2, 7, 3, dtype=torch.double),
        0.1 / 700,
        overlap_detach_widths=True,
        overlap_detach_probabilities=True,
    )
    grads = torch.autograd.grad(metrics["active_overlap"], (means, logits, widths), allow_unused=True)
    assert grads[0].abs().sum() > 0 and grads[1] is None and grads[2] is None


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.paligemma = torch.nn.Linear(8, 8)
        self.model.expert = torch.nn.Linear(8, 8)
        self.model.ctp_parameter_head = head()

    def forward(self, x):
        return self.model.ctp_parameter_head(self.model.expert(self.model.paligemma(x))).square().mean()


def optimizer(model):
    groups = parameter_groups(model.named_parameters(), 1e-5, 0.25, 5, logits_scale=1, width_scale=0.5)
    opt = torch.optim.AdamW(groups, lr=1e-5, betas=(0.9, 0.95), weight_decay=0.01)
    sched = CosineDecayWithWarmupSchedulerConfig(500, 20000, 1e-5, 1e-6).build(opt, 20000)
    return opt, sched


def test_five_group_native_dcp_resume_matches_continuous_step(tmp_path):
    torch.manual_seed(42)
    model = Tiny()
    opt, sched = optimizer(model)
    for _ in range(3):
        model(torch.ones(2, 8)).backward()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
    dcp.save({"optimizer": get_optimizer_state_dict(model, opt)}, checkpoint_id=tmp_path)
    resumed = copy.deepcopy(model)
    new_opt, new_sched = optimizer(resumed)
    load_optimizer_with_lazy_states(resumed, new_opt, tmp_path, StateDictOptions(), set())
    new_sched.load_state_dict(sched.state_dict())
    for net, optim, schedule in ((model, opt, sched), (resumed, new_opt, new_sched)):
        net(torch.ones(2, 8)).backward()
        optim.step()
        schedule.step()
    for a, b in zip(model.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert sched.base_lrs == [2.5e-6, 1e-5, 5e-5, 1e-5, 5e-6]
    assert new_sched.get_last_lr() == sched.get_last_lr()
    assert len(new_opt.state) == len(list(resumed.parameters()))
    assert responsibility_temperature(0) == 20
    assert responsibility_temperature(750) == pytest.approx(20**0.5)
    assert responsibility_temperature(1500) == pytest.approx(1)
    assert responsibility_temperature(20000) == pytest.approx(1)


@pytest.mark.parametrize("workers", [4, 8])
def test_full_validation_exact_12883_frames_with_global_batch_32(workers):
    data = [{"episode_index": i % 17, "value": float(i)} for i in range(12883)]
    padded = PaddedValidationDataset(data, 32)
    assert len(padded) == 12896
    collected = []
    for start in range(0, len(padded), 32):
        for rank in range(workers):
            for i in range(start + rank * (32 // workers), start + (rank + 1) * (32 // workers)):
                row = padded[i]
                if row["ctp_eval_weight"]:
                    collected.append(row["value"])
    assert sorted(collected) == list(range(12883))
    totals = {}
    accumulate_metrics(
        totals,
        {
            "top1_mse": torch.tensor([1.0, 5.0, 999.0]),
            "prefix10_mse": torch.tensor([2.0, 4.0, 999.0]),
            "valid_steps": torch.tensor([50.0, 2.0, 50.0]),
            "prefix_steps": torch.tensor([10.0, 2.0, 10.0]),
        },
        torch.tensor([1.0, 1.0, 0.0]),
        torch.tensor([151, 151, 151]),
        [151],
    )
    assert float(totals["ctp_top1_mse"][0] / totals["ctp_top1_mse"][1]) == pytest.approx(60 / 52)
    assert float(totals["ctp_prefix10_mse"][0] / totals["ctp_prefix10_mse"][1]) == pytest.approx(28 / 12)
    assert totals["samples"][0] == 2


def test_jitter_resumes_by_global_step_without_touching_rng():
    x = torch.rand(2, 3, 8, 8)
    rng = torch.get_rng_state().clone()
    first = color_jitter({"camera": x}, ["camera"], 0.1, 1042)["camera"]
    resumed = color_jitter({"camera": x}, ["camera"], 0.1, 1042)["camera"]
    torch.testing.assert_close(first, resumed, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    assert first.shape == x.shape and first.min() >= 0 and first.max() <= 1
    assert not torch.equal(first, color_jitter({"camera": x}, ["camera"], 0.1, 1043)["camera"])


def test_keep_all_still_marks_complete_and_preserves_prior(tmp_path):
    for step in (2000, 4000):
        folder = tmp_path / str(step)
        (folder / "pretrained_model").mkdir(parents=True)
        (folder / "training_state").mkdir()
        (folder / "pretrained_model/config.json").write_text("{}")
        (folder / "training_state/training_step.json").write_text(json.dumps({"step": step}))
        mark_complete_and_prune(folder, keep=0)
    assert all((tmp_path / str(step) / MARKER).is_file() for step in (2000, 4000))
