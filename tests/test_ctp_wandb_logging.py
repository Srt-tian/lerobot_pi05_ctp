"""Exercise all CTP training and validation namespaces through the real logger."""

from unittest.mock import Mock

import pytest

from lerobot.common.wandb_utils import WandBLogger


@pytest.mark.parametrize("mode", ["train", "eval", "full_eval"])
def test_ctp_validation_namespace_logging(mode):
    logger = WandBLogger.__new__(WandBLogger)
    logger._wandb = Mock()
    logger.log_dict({"samples": 17, "ctp_top1_mse": 0.2}, step=100, mode=mode)
    logger._wandb.log.assert_called_once_with(data={f"{mode}/samples": 17, f"{mode}/ctp_top1_mse": 0.2}, step=100)
