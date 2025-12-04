# clx_mvp/metrics.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence
import torch
from torch.utils.data import DataLoader
from lightning.fabric import Fabric
from torch.utils.flop_counter import FlopCounterMode
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

# ---- Efficiency-aware metrics ----

def _tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()

def count_trainable_params(model: torch.nn.Module) -> int:
    """Number of parameters that require gradients."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_model_memory_bytes(
    model: torch.nn.Module,
    *,
    include_buffers: bool = True,
    include_gradients: bool = False,
    trainable_only: bool = False,
) -> int:
    """
    Estimate model memory footprint (parameters + optional buffers/gradients) in bytes.
    """
    total = 0
    for p in model.parameters():
        if trainable_only and not p.requires_grad:
            continue
        total += _tensor_nbytes(p)
        if include_gradients and p.grad is not None:
            total += _tensor_nbytes(p.grad)

    if include_buffers:
        for b in model.buffers():
            total += _tensor_nbytes(b)

    return total

def estimate_buffer_memory_bytes(buffer: Any) -> int:
    """
    Estimate memory footprint of replay buffers (CPU tensors only).
    Supports ERBuffer and RichERBuffer; falls back to 0 if unsupported.
    """
    total = 0

    if hasattr(buffer, "_x") and hasattr(buffer, "_y"):
        xs = getattr(buffer, "_x", [])
        ys = getattr(buffer, "_y", [])
        total += sum(_tensor_nbytes(t) for t in xs if isinstance(t, torch.Tensor))
        total += sum(_tensor_nbytes(t) for t in ys if isinstance(t, torch.Tensor))
        return total

    if hasattr(buffer, "_entries"):
        entries = getattr(buffer, "_entries", [])
        for ent in entries:
            x = ent.get("x")
            y = ent.get("y")
            if isinstance(x, torch.Tensor):
                total += _tensor_nbytes(x)
            if isinstance(y, torch.Tensor):
                total += _tensor_nbytes(y)
        return total

    return 0

def estimate_flops(
    model: torch.nn.Module,
    input_shape: Sequence[int],
    *,
    include_backward: bool = True,
    device: Optional[torch.device] = None,
) -> Optional[int]:
    """
    Estimate FLOPs for a single training update using Torch's flop counter.

    Args:
        model: nn.Module to profile.
        input_shape: shape tuple for a single batch (e.g., (1, 3, 32, 32)).
        include_backward: if True, doubles forward FLOPs to approximate backward pass.
        device: optional device for the dummy input; defaults to model parameters' device.

    Returns:
        int FLOPs for forward (or forward+backward) pass, or None if estimation fails.
    """
    was_training = model.training
    dev = device
    if dev is None:
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")

    try:
        dummy = torch.zeros(tuple(input_shape), device=dev)
        model.eval()
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            _ = model(dummy)
        fwd_flops = int(counter.get_total_flops())
        total_flops = fwd_flops * 2 if include_backward else fwd_flops
        return total_flops
    except Exception:
        return None
    finally:
        model.train(was_training)

def summarize_efficiency(
    stats_log: Sequence[Dict[str, Any]],
    *,
    model: Optional[torch.nn.Module] = None,
    buffer: Any = None,
    input_shape: Optional[Sequence[int]] = None,
    include_backward_in_flops: bool = True,
) -> Dict[str, Any]:
    """
    Summarize efficiency metrics (time, updates, memory, FLOPs).

    Args:
        stats_log: sequence of per-experience stats dicts emitted by Learner._train_one_experience().
        model: optional model to compute parameter counts/memory/FLOPs.
        buffer: optional replay buffer to measure memory footprint.
        input_shape: shape used to estimate per-forward FLOPs (e.g., (1, 3, 32, 32)).
        include_backward_in_flops: if True, FLOPs include an approximate backward pass.

    Returns:
        Dict with totals and estimates; missing estimates use None.
    """
    total_updates = sum(int(s.get("num_updates", 0)) for s in stats_log)
    total_time = sum(float(s.get("train_time_sec", 0.0)) for s in stats_log)
    max_buffer_size = max((int(s.get("buffer_size", 0)) for s in stats_log), default=0)

    updates_per_sec = total_updates / total_time if total_time > 0 else 0.0
    time_per_update = total_time / total_updates if total_updates > 0 else 0.0

    summary: Dict[str, Any] = {
        "total_updates": total_updates,
        "total_train_time_sec": total_time,
        "updates_per_sec": updates_per_sec,
        "time_per_update_sec": time_per_update,
        "max_buffer_size": max_buffer_size,
    }

    if model is not None:
        summary["trainable_params"] = count_trainable_params(model)
        summary["model_memory_bytes"] = estimate_model_memory_bytes(model)

    if buffer is not None:
        summary["buffer_memory_bytes"] = estimate_buffer_memory_bytes(buffer)

    if model is not None and input_shape is not None:
        flops = estimate_flops(model, input_shape, include_backward=include_backward_in_flops)
        summary["flops_per_update"] = flops
        summary["total_flops"] = flops * total_updates if flops is not None else None

    return summary
