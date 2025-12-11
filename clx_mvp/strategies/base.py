from __future__ import annotations
from abc import ABC
from typing import Tuple
import torch
from torch import Tensor


class CLStrategy(ABC):
    """
    Hook-based interface used by Learner to customize training behavior.
    """
    handles_optimization: bool = False

    def before_experience(self, learner, exp) -> None:
        pass

    def after_experience(self, learner, exp) -> None:
        pass

    def before_batch(self, learner, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Called before each optimizer step.
        Can modify (x, y) in-place (e.g. by concatenating replay).
        """
        return x, y

    def loss(self, learner, logits: Tensor, y: Tensor) -> Tensor:
        """
        Compute loss given logits and labels. Default: CE.
        """
        return learner.criterion(logits, y)

    def after_batch(self, learner, x: Tensor, y: Tensor, loss: torch.Tensor | None = None) -> None:
        """
        Optional hook after optimization.
        """
        return
