from __future__ import annotations
from typing import Optional, Tuple
import copy

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import CLStrategy


class LwFStrategy(CLStrategy):
    """
    Learning without Forgetting: distill into current model from a frozen teacher of previous state.
    Can wrap a base strategy (e.g., ER).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        temperature: float = 2.0,
        base_strategy: Optional[CLStrategy] = None,
    ):
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.base = base_strategy
        self._teacher = None

    def before_experience(self, learner, exp) -> None:
        # snapshot teacher before training this experience
        self._teacher = copy.deepcopy(learner.model)
        for p in self._teacher.parameters():
            p.requires_grad_(False)
        self._teacher.eval()
        if self.base:
            self.base.before_experience(learner, exp)

    def before_batch(self, learner, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        if self.base:
            x, y = self.base.before_batch(learner, x, y)
        else:
            learner._current_batch_for_buffer = (x.detach(), y.detach())

        if self._teacher is not None:
            with torch.no_grad():
                learner._lwf_teacher_logits = self._teacher(x)
        else:
            learner._lwf_teacher_logits = None
        return x, y

    def loss(self, learner, logits: Tensor, y: Tensor) -> Tensor:
        if self.base:
            base_loss = self.base.loss(learner, logits, y)
        else:
            base_loss = learner.criterion(logits, y)

        teacher_logits = getattr(learner, "_lwf_teacher_logits", None)
        if teacher_logits is None:
            return base_loss

        T = self.temperature
        student_log_probs = F.log_softmax(logits / T, dim=1)
        teacher_probs = F.softmax(teacher_logits / T, dim=1)
        distill = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T * T)
        return base_loss + self.alpha * distill

    def after_experience(self, learner, exp) -> None:
        if self.base:
            self.base.after_experience(learner, exp)
        learner._lwf_teacher_logits = None
