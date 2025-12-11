from __future__ import annotations
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .base import CLStrategy


class EWCStrategy(CLStrategy):
    """
    Elastic Weight Consolidation (diagonal Fisher) with optional base strategy (e.g., ER).
    """

    def __init__(
        self,
        lambda_: float = 1000.0,
        fisher_samples: Optional[int] = 1024,
        base_strategy: Optional[CLStrategy] = None,
    ):
        self.lambda_ = float(lambda_)
        self.fisher_samples = fisher_samples
        self.base = base_strategy
        self._prev_params: List[Tensor] = []
        self._fisher: List[Tensor] = []

    def before_experience(self, learner, exp) -> None:
        if self.base:
            self.base.before_experience(learner, exp)

    def before_batch(self, learner, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        if self.base:
            x, y = self.base.before_batch(learner, x, y)
        else:
            learner._current_batch_for_buffer = (x.detach(), y.detach())
        return x, y

    def loss(self, learner, logits: Tensor, y: Tensor) -> Tensor:
        if self.base:
            base_loss = self.base.loss(learner, logits, y)
        else:
            base_loss = learner.criterion(logits, y)

        if not self._prev_params or not self._fisher:
            return base_loss

        penalty = 0.0
        for p, p_old, f in zip(learner.model.parameters(), self._prev_params, self._fisher):
            penalty = penalty + (f * (p - p_old) ** 2).sum()
        return base_loss + 0.5 * self.lambda_ * penalty

    def after_experience(self, learner, exp) -> None:
        self._compute_fisher(learner, exp)
        if self.base:
            self.base.after_experience(learner, exp)

    def _compute_fisher(self, learner, exp) -> None:
        # approximate diagonal Fisher on the current experience
        loader = DataLoader(
            exp.train_ds,
            batch_size=learner.batch_size,
            shuffle=True,
            num_workers=learner.num_workers,
            pin_memory=learner.pin_memory,
        )
        loader = learner.fabric.setup_dataloaders(loader)

        fisher: List[Tensor] = [torch.zeros_like(p) for p in learner.model.parameters()]
        batches = 0

        for x, y in loader:
            learner.optimizer.zero_grad(set_to_none=True)
            logits = learner.model(x)
            log_probs = F.log_softmax(logits, dim=1)
            loss = F.nll_loss(log_probs, y)
            learner.fabric.backward(loss)
            for i, p in enumerate(learner.model.parameters()):
                if p.grad is not None:
                    fisher[i] += p.grad.detach() ** 2
            batches += 1
            if self.fisher_samples is not None and batches >= self.fisher_samples:
                break

        if batches > 0:
            fisher = [f / batches for f in fisher]
        self._fisher = [f.detach() for f in fisher]
        self._prev_params = [p.detach().clone() for p in learner.model.parameters()]
