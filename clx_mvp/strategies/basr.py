from __future__ import annotations
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import CLStrategy
from ..replay import RichERBuffer


class BASRStrategy(CLStrategy):
    """
    Balanced/Adaptive Sample Replay strategy using RichERBuffer.
    Maintains per-sample scores (e.g., current loss) for importance-aware storage/sampling.
    """

    def __init__(
        self,
        replay_ratio: float = 0.5,
        class_balance: bool = True,
        importance_sampling: bool = True,
    ):
        self.replay_ratio = float(replay_ratio)
        self.class_balance = bool(class_balance)
        self.importance_sampling = bool(importance_sampling)

    def _buffer(self, learner) -> RichERBuffer:
        if not isinstance(learner.buffer, RichERBuffer):
            raise ValueError("BASRStrategy requires learner.buffer to be a RichERBuffer.")
        return learner.buffer

    def before_experience(self, learner, exp) -> None:
        pass

    def before_batch(self, learner, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        buffer = self._buffer(learner)
        cur_x, cur_y = x, y

        r_k = int(self.replay_ratio * cur_x.size(0))
        rx, ry = buffer.sample(
            r_k,
            class_balance=self.class_balance,
            importance_sampling=self.importance_sampling,
        )
        if rx is not None:
            rx = rx.to(cur_x.device, non_blocking=True)
            ry = ry.to(cur_y.device, non_blocking=True)
            x = torch.cat([cur_x, rx], dim=0)
            y = torch.cat([cur_y, ry], dim=0)

        learner._current_batch_for_buffer = (cur_x.detach(), cur_y.detach(), None)
        learner._basr_current_size = cur_x.size(0)

        return x, y

    def loss(self, learner, logits: Tensor, y: Tensor) -> Tensor:
        base_loss = learner.criterion(logits, y)

        cur_size = getattr(learner, "_basr_current_size", y.size(0))
        if cur_size > 0:
            per_sample = F.cross_entropy(logits, y, reduction="none")
            scores = per_sample[:cur_size].detach()

            cur_batch = getattr(learner, "_current_batch_for_buffer", None)
            if cur_batch is not None and isinstance(cur_batch, tuple) and len(cur_batch) >= 2:
                cur_x, cur_y = cur_batch[0], cur_batch[1]
                learner._current_batch_for_buffer = (cur_x, cur_y, scores)

        return base_loss

    def after_experience(self, learner, exp) -> None:
        learner._basr_current_size = 0
