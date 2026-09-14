import pytest
import torch
from safetensors.torch import save_file

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


class TinyLoadingPolicy(PI05Policy):
    """Exercise the real loader without allocating the multi-billion-parameter graph."""

    def __init__(self, config, **kwargs):
        torch.nn.Module.__init__(self)
        self.config = config
        self.model = torch.nn.Module()
        self.model.backbone = torch.nn.Linear(2, 2)
        self.model.ctp_parameter_head = torch.nn.Linear(2, 2)


def config():
    return PI05Config(ctp_enabled=True, ctp_belief_cumulative_action_dims=0, device="cpu")


def test_missing_weights_cannot_silently_return_random_model(tmp_path):
    with pytest.raises(RuntimeError, match="Failed to load"):
        TinyLoadingPolicy.from_pretrained(tmp_path, config=config(), local_files_only=True)


def test_new_ctp_head_is_the_only_permitted_missing_module(tmp_path):
    source = TinyLoadingPolicy(config())
    weights = {k: v for k, v in source.state_dict().items() if "ctp_parameter_head" not in k}
    save_file(weights, tmp_path / "model.safetensors")
    loaded = TinyLoadingPolicy.from_pretrained(tmp_path, config=config())
    torch.testing.assert_close(loaded.model.backbone.weight, source.model.backbone.weight)
    weights.pop("model.backbone.bias")
    save_file(weights, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="Failed to load"):
        TinyLoadingPolicy.from_pretrained(tmp_path, config=config())


def test_ctp_disabled_preserves_small_action_baseline_configuration():
    cfg = PI05Config(
        ctp_enabled=False, output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))}
    )
    assert not cfg.ctp_enabled
