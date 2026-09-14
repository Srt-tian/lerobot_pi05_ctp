"""Capacity-preserving split CTP head and step-derived responsibility schedule."""

import math

import torch
from torch import nn


def responsibility_temperature(step, initial=20.0, final=1.0, anneal_steps=1500):
    if step < 0 or anneal_steps <= 0 or min(initial, final) <= 0:
        raise ValueError("Invalid responsibility schedule")
    fraction = min(step / anneal_steps, 1.0)
    return initial * (final / initial) ** fraction


class SplitCTPHead(nn.Module):
    """Same affine parameter count/layout on output as a fused legacy head.

    Width-only gradients stop at the context; center/logit paths still train
    the full backbone. No feature detachment is applied to the other branches.
    """

    def __init__(
        self,
        context_dim,
        modes,
        trajectory_dim,
        init_std,
        min_width,
        initial_width,
        max_width,
        detach_width_context=True,
    ):
        super().__init__()
        self.modes = modes
        self.trajectory_dim = trajectory_dim
        self.detach_width_context = detach_width_context
        self.centers = nn.Linear(context_dim, modes * trajectory_dim)
        self.logits = nn.Linear(context_dim, modes)
        self.widths = nn.Linear(context_dim, modes)
        for layer in (self.centers, self.logits, self.widths):
            nn.init.normal_(layer.weight, std=init_std)
            nn.init.zeros_(layer.bias)
        fraction = (initial_width - min_width) / (max_width - min_width)
        nn.init.constant_(self.widths.bias, math.log(fraction / (1 - fraction)))

    def forward(self, context):
        centers = self.centers(context).reshape(-1, self.modes, self.trajectory_dim)
        logits = self.logits(context).unsqueeze(-1)
        widths = self.widths(context.detach() if self.detach_width_context else context).unsqueeze(-1)
        return torch.cat((centers, logits, widths), dim=-1).flatten(1)
