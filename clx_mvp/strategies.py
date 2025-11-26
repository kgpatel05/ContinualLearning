# clx_mvp/strategies.py
from __future__ import annotations
from abc import ABC
from typing import Optional, Tuple, List
import copy

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .replay import ERBuffer, RichERBuffer


class CLStrategy(ABC):
    """
    Hook-based interface used by Learner to customize training behavior.
    """

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


class FinetuneStrategy(CLStrategy):
    """
    Finetuning: no replay, vanilla CE.
    """

    def before_batch(self, learner, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        # no replay; skip buffer admission
        learner._current_batch_for_buffer = None
        return x, y


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


class AGEMStrategy(CLStrategy):
    """
    Averaged GEM (A-GEM) with gradient projection using episodic memory (ERBuffer).
    """

    def __init__(
        self,
        mem_batch_size: int = 256,
        base_strategy: Optional[CLStrategy] = None,
    ):
        self.mem_batch_size = int(mem_batch_size)
        self.base = base_strategy

    def before_experience(self, learner, exp) -> None:
        if self.base:
            self.base.before_experience(learner, exp)

    def before_batch(self, learner, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        if self.base:
            x, y = self.base.before_batch(learner, x, y)
        else:
            learner._current_batch_for_buffer = (x.detach(), y.detach())

        mem_x, mem_y = learner.buffer.sample(self.mem_batch_size)
        if mem_x is not None:
            mem_x = mem_x.to(x.device, non_blocking=True)
            mem_y = mem_y.to(y.device, non_blocking=True)
            learner._agem_ref_batch = (mem_x, mem_y)
        else:
            learner._agem_ref_batch = None
        return x, y

    def loss(self, learner, logits: Tensor, y: Tensor) -> Tensor:
        params = [p for p in learner.model.parameters() if p.requires_grad]

        if self.base:
            base_loss = self.base.loss(learner, logits, y)
        else:
            base_loss = learner.criterion(logits, y)

        g_cur = torch.autograd.grad(base_loss, params, retain_graph=True, allow_unused=True)
        g_cur = [gc if gc is not None else torch.zeros_like(p) for gc, p in zip(g_cur, params)]

        ref_batch = getattr(learner, "_agem_ref_batch", None)
        if ref_batch is not None:
            mem_x, mem_y = ref_batch
            ref_logits = learner.model(mem_x)
            ref_loss = learner.criterion(ref_logits, mem_y)
            g_ref = torch.autograd.grad(ref_loss, params, retain_graph=False, allow_unused=True)
            g_ref = [gr if gr is not None else torch.zeros_like(p) for gr, p in zip(g_ref, params)]
        else:
            g_ref = [torch.zeros_like(p) for p in params]

        dot = sum((gc * gr).sum() for gc, gr in zip(g_cur, g_ref))
        ref_norm = sum((gr ** 2).sum() for gr in g_ref) + 1e-12

        if ref_batch is not None and dot < 0:
            proj_scale = dot / ref_norm
            g_new = [gc - proj_scale * gr for gc, gr in zip(g_cur, g_ref)]
        else:
            g_new = g_cur

        surrogate = 0.0
        for p, g in zip(params, g_new):
            surrogate = surrogate + (p * g.detach()).sum()
        return surrogate

    def after_experience(self, learner, exp) -> None:
        if self.base:
            self.base.after_experience(learner, exp)
        learner._agem_ref_batch = None


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
