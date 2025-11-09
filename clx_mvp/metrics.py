# clx_mvp/metrics.py
from __future__ import annotations
from typing import List, Sequence, Dict
import torch
from torch.utils.data import DataLoader
from lightning.fabric import Fabric

@torch.no_grad()
def accuracy(model, loader: DataLoader, fabric: Fabric) -> float:
    """
    Compute top-1 accuracy (%).

    Inputs:
        model: nn.Module in eval() or train(); will be used under torch.no_grad
        loader: DataLoader yielding (x, y)
        fabric: Lightning Fabric instance (used for device placement)

    Returns:
        float: accuracy in [0, 100].
    """
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = fabric.to_device((x, y))
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(1, total)

def average_accuracy(acc_after_exp: Sequence[float]) -> float:
    """
    Average Accuracy (AA) over experiences.

    Inputs:
        acc_after_exp: list/sequence where acc_after_exp[k] is accuracy after training experience k

    Returns:
        float: mean of the sequence (0..100)
    """
    n = len(acc_after_exp)
    if n == 0:
        return 0.0
    return float(sum(acc_after_exp) / n)

def compute_forgetting(acc_matrix: List[List[float]]) -> List[float]:
    """
    Compute forgetting per experience from an accuracy matrix.

    Inputs:
        acc_matrix: 2D list where acc_matrix[i][t] is accuracy on experience i
                    measured *after training experience t* (i <= t).
                    For MVP you may only fill diagonal entries; full forgetting
                    requires evaluating past exps after each new one.

    Returns:
        List[float]: forgetting[i] = max_{t>=i} acc_matrix[i][t] - acc_matrix[i][T],
                     where T = last training experience index.
                     If only diagonal is available, returns zeros (no info).
    """
    if not acc_matrix:
        return []

    T = len(acc_matrix) - 1
    forgetting = []
    for i, row in enumerate(acc_matrix):
        # row length can vary; consider values from i..len(row)-1
        tail = row[i:] if len(row) > i else []
        if len(tail) == 0:
            forgetting.append(0.0)
        else:
            best = max(tail)
            final = tail[-1]
            forgetting.append(max(0.0, best - final))
    return forgetting
