from __future__ import annotations
from typing import Tuple

import torch
from torch import Tensor

from .base import CLStrategy
from ..replay import ERBuffer


class ERStrategy(CLStrategy):
    """
    Reproduces the current behavior: ER with reservoir buffer.
    """

    def __init__(self, replay_ratio: float = 0.5):
        self.replay_ratio = float(replay_ratio)

    def before_batch(self, learner, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        buffer: ERBuffer = learner.buffer
        r_k = int(self.replay_ratio * x.size(0))
        rx, ry = buffer.sample(r_k)
        if rx is not None:
            rx = rx.to(x.device, non_blocking=True)
            ry = ry.to(y.device, non_blocking=True)
            cur_x, cur_y = x, y
            x = torch.cat([cur_x, rx], dim=0)
            y = torch.cat([cur_y, ry], dim=0)
            # stash current part so we only add those back
            learner._current_batch_for_buffer = (cur_x.detach(), cur_y.detach())
        else:
            learner._current_batch_for_buffer = (x.detach(), y.detach())
        return x, y

    def after_experience(self, learner, exp) -> None:
        # nothing special here; per-batch admission already done
        pass
