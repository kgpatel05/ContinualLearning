from __future__ import annotations
from typing import Optional, Tuple, List

import torch
from torch import Tensor

from .base import CLStrategy


class AGEMStrategy(CLStrategy):
    """
    Averaged GEM (A-GEM) with gradient projection using episodic memory (ERBuffer).
    
    Uses fabric.backward() to compute gradients, then manually projects them
    to avoid conflicts with reference gradients from memory.
    """

    def __init__(
        self,
        mem_batch_size: int = 256,
        base_strategy: Optional[CLStrategy] = None,
    ):
        self.mem_batch_size = int(mem_batch_size)
        self.base = base_strategy
        self._skip_backward = False  # Flag to signal custom gradient handling

    def before_experience(self, learner, exp) -> None:
        if self.base:
            self.base.before_experience(learner, exp)

    def before_batch(self, learner, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        if self.base:
            x, y = self.base.before_batch(learner, x, y)
        else:
            learner._current_batch_for_buffer = (x.detach(), y.detach())

        # Sample reference batch from memory
        mem_x, mem_y = learner.buffer.sample(self.mem_batch_size)
        if mem_x is not None:
            mem_x = mem_x.to(x.device, non_blocking=True)
            mem_y = mem_y.to(y.device, non_blocking=True)
            learner._agem_ref_batch = (mem_x, mem_y)
        else:
            learner._agem_ref_batch = None
        
        self._skip_backward = True  # Signal to Learner to skip normal backward
        return x, y

    def loss(self, learner, logits: Tensor, y: Tensor) -> Tensor:
        """
        Compute current loss (will be used for backward in apply_agem_gradients).
        """
        if self.base:
            return self.base.loss(learner, logits, y)
        else:
            return learner.criterion(logits, y)

    def apply_agem_gradients(self, learner, current_loss: Tensor) -> None:
        """
        Custom gradient computation and projection for AGEM.
        
        Steps:
        1. Backward on current loss to get g_cur
        2. Copy current gradients
        3. Zero gradients and backward on reference loss to get g_ref
        4. Project g_cur if it conflicts with g_ref (dot product < 0)
        5. Set projected gradients back to parameters
        """
        params = [p for p in learner.model.parameters() if p.requires_grad]
        
        # Step 1: Compute current gradients
        learner.optimizer.zero_grad(set_to_none=True)
        learner.fabric.backward(current_loss)
        
        # Step 2: Copy current gradients
        g_cur = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        
        # Step 3: Compute reference gradients
        ref_batch = getattr(learner, "_agem_ref_batch", None)
        if ref_batch is not None:
            mem_x, mem_y = ref_batch
            learner.optimizer.zero_grad(set_to_none=True)
            
            ref_logits = learner.model(mem_x)
            ref_loss = learner.criterion(ref_logits, mem_y)
            learner.fabric.backward(ref_loss)
            
            g_ref = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        else:
            g_ref = [torch.zeros_like(p) for p in params]
        
        # Step 4: Project if current gradients conflict with reference gradients
        dot = sum((gc * gr).sum() for gc, gr in zip(g_cur, g_ref))
        
        if ref_batch is not None and dot < 0:
            # Negative dot product: project g_cur onto g_ref
            ref_norm = sum((gr ** 2).sum() for gr in g_ref) + 1e-12
            proj_scale = dot / ref_norm
            g_new = [gc - proj_scale * gr for gc, gr in zip(g_cur, g_ref)]
        else:
            # No conflict or no reference: use current gradients as-is
            g_new = g_cur
        
        # Step 5: Set projected gradients
        for p, g in zip(params, g_new):
            p.grad = g

    def after_experience(self, learner, exp) -> None:
        if self.base:
            self.base.after_experience(learner, exp)
        learner._agem_ref_batch = None
