# clx_mvp/metrics.py
from __future__ import annotations
from typing import List, Sequence
import torch
from torch.utils.data import DataLoader
from lightning.fabric import Fabric
from .streams import Experience


class ContinualEvaluator:
    """
    Utility to compute the accuracy matrix over a stream of experiences.

    Usage:
        evaluator = ContinualEvaluator(stream, fabric, batch_size=128)
        for k, exp in enumerate(stream):
            # after training on exp
            row = evaluator.evaluate_after_exp(model, upto_exp=k)
            ...
    """
    def __init__(self, stream: Sequence[Experience], fabric: Fabric, batch_size: int = 128, num_workers: int = 2):
        self.stream = list(stream)
        self.fabric = fabric
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.acc_matrix: List[List[float]] = []

    def _make_loader(self, ds) -> DataLoader:
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return self.fabric.setup_dataloaders(loader)

    @torch.no_grad()
    def evaluate_after_exp(self, model, upto_exp: int) -> List[float]:
        """
        Evaluate 'model' on test sets of experiences [0 .. upto_exp].
        Appends a row to acc_matrix and returns it.
        """
        model.eval()
        row: List[float] = []
        for exp in self.stream[: upto_exp + 1]:
            loader = self._make_loader(exp.test_ds)
            row.append(accuracy(model, loader, self.fabric))
        self.acc_matrix.append(row)
        return row

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
    Compute forgetting per task from an accuracy matrix.

    The matrix structure is:
        acc_matrix[i][j] = accuracy on task j after training experience i
        (rows can be ragged: row i has length i+1)

    Forgetting for task j is:
        max_{i>=j} acc_matrix[i][j] - acc_matrix[T][j]
    where T is the final experience index.

    Args:
        acc_matrix: 2D list where acc_matrix[i][j] is accuracy on task j
                    measured after training experience i (j <= i).

    Returns:
        List[float]: forgetting[j] is the forgetting for task j,
                     measured as (peak accuracy - final accuracy).
    """
    if not acc_matrix:
        return []

    n_exps = len(acc_matrix)
    n_tasks = len(acc_matrix[-1])  # number of tasks seen at the end

    forgetting: List[float] = []
    for j in range(n_tasks):
        # Collect all accuracies for task j across experiences where it's evaluated
        acc_j = [acc_matrix[i][j] for i in range(n_exps) if j < len(acc_matrix[i])]
        
        if not acc_j:
            forgetting.append(0.0)
            continue
        
        max_acc = max(acc_j)
        final_acc = acc_j[-1]  # last evaluation of task j
        forgetting.append(max(0.0, max_acc - final_acc))
    
    return forgetting
