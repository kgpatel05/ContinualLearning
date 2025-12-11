from __future__ import annotations
from typing import Tuple
from torch import Tensor

from .base import CLStrategy


class FinetuneStrategy(CLStrategy):
    """
    Finetuning: no replay, vanilla CE.
    """

    def before_batch(self, learner, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        # no replay; skip buffer admission
        learner._current_batch_for_buffer = None
        return x, y
