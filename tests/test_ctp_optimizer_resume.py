import copy

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

from lerobot.distributed.checkpoint import load_optimizer_with_lazy_states


class TwoHeads(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.used = torch.nn.Linear(3, 2)
        self.unused = torch.nn.Linear(3, 2)


def setup_checkpoint(tmp_path, drop_active_state=False):
    torch.manual_seed(12)
    model = TwoHeads()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    x = torch.randn(4, 3)
    model.used(x).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    state = get_optimizer_state_dict(model, optimizer)
    if drop_active_state:
        state["state"].pop("used.bias")
    dcp.save({"optimizer": state}, checkpoint_id=tmp_path)
    return model, optimizer, x


def test_lazy_unused_state_resume_matches_next_uninterrupted_update(tmp_path):
    model, optimizer, x = setup_checkpoint(tmp_path)
    resumed = copy.deepcopy(model)
    new_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.001)
    load_optimizer_with_lazy_states(
        resumed, new_optimizer, tmp_path, StateDictOptions(), {"unused.weight", "unused.bias"}
    )
    assert not new_optimizer.state[resumed.unused.weight]
    assert not new_optimizer.state[resumed.unused.bias]
    for key in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            new_optimizer.state[resumed.used.weight][key], optimizer.state[model.used.weight][key]
        )
    for network, opt in ((model, optimizer), (resumed, new_optimizer)):
        network.used(x).square().mean().backward()
        opt.step()
    for a, b in zip(model.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_missing_active_optimizer_state_still_fails(tmp_path):
    model, _, _ = setup_checkpoint(tmp_path, drop_active_state=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    with pytest.raises(CheckpointException):
        load_optimizer_with_lazy_states(
            model, optimizer, tmp_path, StateDictOptions(), {"unused.weight", "unused.bias"}
        )
