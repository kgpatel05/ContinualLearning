# clx_mvp/lora.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, List, Tuple
import torch
import torch.nn as nn


class LoraLinear(nn.Module):
    """
    Lightweight LoRA adapter wrapping a Linear layer.
    """
    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.rank)

        dev = base.weight.device
        dt = base.weight.dtype
        self.A = nn.Linear(base.in_features, self.rank, bias=False, device=dev, dtype=dt)
        self.B = nn.Linear(self.rank, base.out_features, bias=False, device=dev, dtype=dt)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)

        # Freeze base weights to limit plasticity
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + self.B(self.A(x)) * self.scaling

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias


@dataclass
class LoraConfig:
    rank: int = 4
    alpha: float = 8.0
    target_modules: Optional[List[str]] = None  # match substrings of module names; None => all linear layers


def _should_wrap(name: str, target: Optional[List[str]]) -> bool:
    if target is None or len(target) == 0:
        return True
    return any(tok in name for tok in target)


def inject_lora_layers(model: nn.Module, cfg: LoraConfig) -> List[Tuple[str, LoraLinear]]:
    """
    Replace selected nn.Linear layers with LoRA-wrapped versions.
    Returns the list of (name, LoraLinear) replacements applied.
    """
    replacements: List[Tuple[str, LoraLinear]] = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and _should_wrap(name, cfg.target_modules):
            parent, attr = _resolve_parent(model, name)
            wrapped = LoraLinear(module, rank=cfg.rank, alpha=cfg.alpha)
            setattr(parent, attr, wrapped)
            replacements.append((name, wrapped))
    return replacements


def _resolve_parent(model: nn.Module, name: str) -> Tuple[nn.Module, str]:
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]
