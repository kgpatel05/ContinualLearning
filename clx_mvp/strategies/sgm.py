# clx_mvp/sgm.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Set, List, Tuple
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import CLStrategy
from ..features import FeatureExtractor, FeatureExtractorConfig
from ..lora import LoraConfig, inject_lora_layers


@dataclass
class SgmConfig:
    """
    Configuration for Stability Gap Mitigation wrapper.
    """
    head_attr: str = "fc"

    # Soft targets
    use_soft_targets: bool = True
    soft_target_alpha: float = 0.5
    soft_target_temperature: float = 2.0
    soft_new_class_only: bool = True

    # Data-driven init
    use_weight_init: bool = True
    init_samples_per_class: int = 16
    feature_extractor: FeatureExtractorConfig = field(default_factory=FeatureExtractorConfig)

    # LoRA
    use_lora: bool = True
    lora: LoraConfig = field(default_factory=LoraConfig)

    # Old output class freezing
    use_oocf: bool = True
    oocf_mode: str = "soft"  # "soft" or "hard"
    oocf_strength: float = 1.0
    oocf_steps: int = 200  # apply OOCF penalty for first N steps of new-class training


class SgmStrategy(CLStrategy):
    """
    Wrapper that augments a base strategy with Stability Gap Mitigation components.

    Components:
        - Dynamic soft targets (teacher mixing)
        - Data-driven initialization for new class outputs
        - LoRA adapters to limit hidden-layer plasticity
        - Old output class freezing (hard/soft)

    Reference: "Overcoming the Stability Gap in Continual Learning" (ICLR 2024).
    """
    def __init__(self, cfg: SgmConfig, base_strategy: Optional[CLStrategy] = None):
        self.cfg = cfg
        self.base = base_strategy
        self.extractor = FeatureExtractor(cfg.feature_extractor)
        self._teacher: Optional[nn.Module] = None
        self._seen_classes: Set[int] = set()
        self._current_new_classes: Set[int] = set()
        self._oocf_reference: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]] = None
        self._oocf_steps_remaining: int = 0
        self._lora_applied: bool = False

    # ---- Hooks ----
    def before_experience(self, learner, exp) -> None:
        if self.cfg.use_lora and not self._lora_applied:
            inject_lora_layers(learner.model, self.cfg.lora)
            self._lora_applied = True

        exp_classes = set(getattr(exp, "classes", []))
        self._current_new_classes = {c for c in exp_classes if c not in self._seen_classes}

        if self.cfg.use_weight_init:
            self._data_driven_init(learner, exp)

        if self.cfg.use_soft_targets:
            self._teacher = copy.deepcopy(learner.model).eval()
            for p in self._teacher.parameters():
                p.requires_grad_(False)

        if self.cfg.use_oocf:
            head = self._get_head(learner.model)
            self._oocf_reference = (
                head.weight.detach().clone(),
                None if head.bias is None else head.bias.detach().clone(),
            )
            self._oocf_steps_remaining = self.cfg.oocf_steps

        if self.base:
            self.base.before_experience(learner, exp)

    def before_batch(self, learner, x, y):
        if self.base:
            x, y = self.base.before_batch(learner, x, y)
        if self.cfg.use_soft_targets and self._teacher is not None:
            with torch.no_grad():
                learner._sgm_teacher_logits = self._teacher(x)
        else:
            learner._sgm_teacher_logits = None
        return x, y

    def loss(self, learner, logits, y):
        base_loss = self.base.loss(learner, logits, y) if self.base else learner.criterion(logits, y)
        total = base_loss

        if self.cfg.use_soft_targets and hasattr(learner, "_sgm_teacher_logits"):
            teacher_logits = learner._sgm_teacher_logits
            if teacher_logits is not None:
                total = total + self._soft_target_loss(logits, teacher_logits, y)

        if self.cfg.use_oocf and self._oocf_reference is not None and self.cfg.oocf_mode == "soft":
            total = total + self._oocf_penalty(learner)

        return total

    def after_batch(self, learner, x, y, loss=None):
        if self.cfg.use_oocf and self.cfg.oocf_mode == "hard" and self._oocf_steps_remaining > 0:
            self._apply_hard_oocf(learner)
            self._oocf_steps_remaining -= 1
        if self.base and hasattr(self.base, "after_batch"):
            try:
                self.base.after_batch(learner, x, y, loss)  # type: ignore[arg-type]
            except TypeError:
                self.base.after_batch(learner, x, y)

    def after_experience(self, learner, exp) -> None:
        self._seen_classes.update(getattr(exp, "classes", []))
        self._current_new_classes = set()
        self._teacher = None
        self._oocf_steps_remaining = 0
        if self.base:
            self.base.after_experience(learner, exp)

    # ---- components ----
    def _soft_target_loss(self, logits: torch.Tensor, teacher_logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        T = self.cfg.soft_target_temperature
        teacher_probs = F.softmax(teacher_logits / T, dim=1)
        hard_targets = F.one_hot(y, num_classes=logits.size(1)).float()
        mixed = self.cfg.soft_target_alpha * teacher_probs + (1 - self.cfg.soft_target_alpha) * hard_targets

        if self.cfg.soft_new_class_only and self._current_new_classes:
            mask = torch.isin(y, torch.tensor(list(self._current_new_classes), device=y.device))
            if not mask.any():
                return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            loss = -(mixed * F.log_softmax(logits / T, dim=1)).sum(dim=1)
            loss = (loss * mask.float()).sum() / mask.float().sum()
        else:
            loss = -(mixed * F.log_softmax(logits / T, dim=1)).sum(dim=1).mean()
        return loss

    def _data_driven_init(self, learner, exp) -> None:
        new_classes = list(self._current_new_classes)
        if not new_classes:
            return

        loader = DataLoader(
            exp.train_ds,
            batch_size=max(4, self.cfg.init_samples_per_class),
            shuffle=True,
            num_workers=getattr(learner, "num_workers", 0),
            pin_memory=getattr(learner, "pin_memory", False),
        )
        loader = learner.fabric.setup_dataloaders(loader)

        collected: dict[int, List[torch.Tensor]] = {c: [] for c in new_classes}
        for x_batch, y_batch in loader:
            feats = self.extractor(learner.model, x_batch).detach()
            for cls in new_classes:
                mask = (y_batch == cls)
                if mask.any():
                    collected[cls].append(feats[mask].mean(dim=0))
            if all(len(collected[c]) >= 1 for c in new_classes):
                break

        head = self._get_head(learner.model)
        device = next(head.parameters()).device
        with torch.no_grad():
            for cls in new_classes:
                if not collected[cls]:
                    continue
                mean_feat = torch.stack(collected[cls], dim=0).mean(dim=0).to(device)
                scaled = mean_feat / (mean_feat.norm() + 1e-6)
                head.weight.data[cls] = scaled
                if head.bias is not None:
                    head.bias.data[cls] = 0.0

    def _oocf_penalty(self, learner) -> torch.Tensor:
        if self._oocf_reference is None:
            return torch.tensor(0.0, device=next(learner.model.parameters()).device)
        head = self._get_head(learner.model)
        ref_w, ref_b = self._oocf_reference
        old_classes = sorted(self._seen_classes - self._current_new_classes)
        if not old_classes:
            return torch.tensor(0.0, device=head.weight.device)
        idx = torch.tensor(old_classes, device=head.weight.device)
        penalty = (head.weight[idx] - ref_w[idx]).pow(2).mean()
        if head.bias is not None and ref_b is not None:
            penalty = penalty + (head.bias[idx] - ref_b[idx]).pow(2).mean()
        return self.cfg.oocf_strength * penalty

    def _apply_hard_oocf(self, learner) -> None:
        head = self._get_head(learner.model)
        old_classes = sorted(self._seen_classes - self._current_new_classes)
        if not old_classes:
            return
        idx = torch.tensor(old_classes, device=head.weight.device)
        if head.weight.grad is not None:
            head.weight.grad[idx] = 0
        if head.bias is not None and head.bias.grad is not None:
            head.bias.grad[idx] = 0

    def _get_head(self, model: nn.Module) -> nn.Linear:
        head_attr = self.cfg.head_attr
        head = getattr(model, head_attr, None)
        if head is None or not isinstance(head, nn.Linear):
            raise ValueError(f"SGM expects model to expose a linear head '{head_attr}'")
        return head
