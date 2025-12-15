# clx_mvp/learner.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lightning.fabric import Fabric

from .streams import Experience
from .replay import ERBuffer
from .metrics import accuracy, average_accuracy
from .strategies import CLStrategy, ERStrategy

@dataclass
class FitReport:
    """
    Aggregate results returned by Learner.fit(...)

    Attributes:
        per_exp_acc: list of accuracies (current-experience test) after each exp
        avg_acc: average of per_exp_acc
        checkpoints: list of checkpoint filepaths (if saved)
    """
    per_exp_acc: List[float]
    avg_acc: float
    checkpoints: List[str]

class Learner:
    """
    Minimal continual-learning learner with:
      - Fabric-managed device/precision
      - single-head classifier (CrossEntropy)
      - ER replay with reservoir buffer
      - evaluation on current-experience test set (MVP)

    Public methods:
        fit(stream: List[Experience]) -> FitReport
        save_checkpoint(dirpath: str, exp_id: int) -> str
        load_checkpoint(filepath: str) -> None
    """

    def __init__(
        self,
        model: nn.Module,
        fabric: Fabric,
        buffer: ERBuffer,
        *,
        lr: float = 0.03,
        weight_decay: float = 5e-4,
        momentum: float = 0.9,
        replay_ratio: float = 0.5,
        strategy: Optional[CLStrategy] = None,
        batch_size: int = 128,
        epochs: int = 1,
        num_workers: int = 2,
        pin_memory: bool = True,
        precision: Optional[str] = None,  # if you want to surface here later
    ) -> None:
        """
        Inputs:
            model: nn.Module (e.g., ResNet18 with num_classes=10)
            fabric: Fabric runtime (e.g., Fabric(accelerator="auto", devices=1, precision="bf16-mixed"))
            buffer: ERBuffer instance
            lr, weight_decay, momentum: SGD hyperparameters
            replay_ratio: proportion of each batch to draw from buffer (0..1)
            strategy: CLStrategy implementation; defaults to ERStrategy(replay_ratio)
            batch_size: batch size per DataLoader on this process
            epochs: number of epochs per experience
            num_workers: DataLoader workers
            pin_memory: pass to DataLoader (harmless if unsupported)
            precision: reserved; fabric already configured outside
        """
        self.fabric = fabric
        self.model = model
        self.buffer = buffer

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        # Let Fabric wrap model+optimizer for device/precision
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)

        # training hyperparams
        self.replay_ratio = float(replay_ratio)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.strategy: CLStrategy = strategy or ERStrategy(replay_ratio=replay_ratio)
        self._last_exp_stats: List[Dict[str, Any]] = []

    # ---- public API ----

    def fit(self, stream: List[Experience]) -> FitReport:
        """
        Train sequentially over experiences.

        Inputs:
            stream: ordered list of Experience objects

        Returns:
            FitReport: per-experience accuracies (current-experience test),
                       final average accuracy, and checkpoint filepaths.

        Side effects:
            Prints per-experience logs; saves checkpoints to ./checkpoints by default.
        """
        per_exp_acc: List[float] = []
        checkpoints: List[str] = []

        for exp in stream:
            self.strategy.before_experience(self, exp)
            tr_loader, te_loader = self._make_loaders(exp)
            stats = self._train_one_experience(tr_loader, exp_id=exp.exp_id)
            self.strategy.after_experience(self, exp)
            acc = accuracy(self.model, te_loader, self.fabric)
            per_exp_acc.append(acc)
            self.fabric.print(f"[Exp {exp.exp_id}] classes={exp.classes} acc={acc:.2f}% AA={average_accuracy(per_exp_acc):.2f}%")

            # checkpoint after each experience (optional dir)
            ckpt_path = self.save_checkpoint("checkpoints", exp.exp_id)
            if ckpt_path:
                checkpoints.append(ckpt_path)

        return FitReport(per_exp_acc=per_exp_acc, avg_acc=average_accuracy(per_exp_acc), checkpoints=checkpoints)

    def save_checkpoint(self, dirpath: str, exp_id: int) -> str:
        """
        Save model/optimizer/buffer state after an experience.

        Inputs:
            dirpath: directory path where to save files (created if missing)
            exp_id: used to name the checkpoint file

        Returns:
            str: path to saved checkpoint (only on rank 0); empty string otherwise
        """
        if self.fabric.global_rank != 0:
            return ""
        os.makedirs(dirpath, exist_ok=True)
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "buffer": self.buffer.state_dict(),
            "exp_id": exp_id,
        }
        path = os.path.join(dirpath, f"exp{exp_id}.pt")
        torch.save(state, path)
        return path

    def load_checkpoint(self, filepath: str) -> None:
        """
        Restore model/optimizer/buffer.

        Inputs:
            filepath: path returned by save_checkpoint()

        Side effects:
            Overwrites model/optimizer/buffer state in-place.
        """
        map_location = self.fabric.device if hasattr(self.fabric, "device") else "cpu"
        state = torch.load(filepath, map_location=map_location)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.buffer.load_state_dict(state["buffer"])

    # ---- internal helpers ----

    def _make_loaders(self, exp: Experience) -> tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(
            exp.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        test_loader = DataLoader(
            exp.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        # let Fabric set up device, etc.
        train_loader = self.fabric.setup_dataloaders(train_loader)
        test_loader = self.fabric.setup_dataloaders(test_loader)
        return train_loader, test_loader

    def _train_one_experience(self, train_loader: DataLoader, exp_id: Optional[int] = None) -> dict[str, Any]:
        """
        One training block for the current experience.
        Returns a dict with efficiency stats.

        Behavior:
            - Delegates per-batch augmentation/replay to strategy hooks.
            - Computes loss via strategy, backward, step.
            - Special handling for AGEM: uses custom gradient projection.
            - Admits only the current batch samples into the buffer.
        """
        self.model.train()
        num_updates = 0
        t0 = time.perf_counter()

        for _ in range(self.epochs):
            for x, y in train_loader:
                x, y = self.strategy.before_batch(self, x, y)
                # cache current batch for strategies that may need to recompute forward/backward
                self._current_batch_inputs = (x, y)

                logits = self.model(x)
                loss = self.strategy.loss(self, logits, y)
                handled = False
                if getattr(self.strategy, "handles_optimization", False) and hasattr(self.strategy, "optimize_batch"):
                    # Strategy owns backward/step (e.g., SIESTA sleep/wake control)
                    self.strategy.optimize_batch(self, loss, logits, y)  # type: ignore[arg-type]
                    handled = True
                elif hasattr(self.strategy, '_skip_backward') and self.strategy._skip_backward:
                    # AGEM uses custom backward/projection
                    if hasattr(self.strategy, 'apply_agem_gradients'):
                        self.strategy.apply_agem_gradients(self, loss)
                        self.optimizer.step()
                    else:
                        # Fallback to normal backward if method missing
                        self.optimizer.zero_grad(set_to_none=True)
                        self.fabric.backward(loss)
                        self.optimizer.step()
                    handled = True

                if not handled:
                    # Normal backward pass for other strategies
                    self.optimizer.zero_grad(set_to_none=True)
                    self.fabric.backward(loss)
                    self.optimizer.step()
                
                num_updates += 1

                skip_buffer_add = getattr(self, "_skip_buffer_addition", False)
                if not skip_buffer_add:
                    cur_batch = getattr(self, "_current_batch_for_buffer", None)
                    if cur_batch is None:
                        cur_batch = (x.detach(), y.detach())

                    if isinstance(cur_batch, tuple) and len(cur_batch) == 4:
                        cur_x, cur_y, scores, feats = cur_batch
                        try:
                            self.buffer.add(cur_x, cur_y, scores=scores, features=feats)
                        except TypeError:
                            self.buffer.add(cur_x, cur_y)
                    elif isinstance(cur_batch, tuple) and len(cur_batch) == 3:
                        cur_x, cur_y, scores = cur_batch
                        try:
                            self.buffer.add(cur_x, cur_y, scores=scores)
                        except TypeError:
                            self.buffer.add(cur_x, cur_y)
                    else:
                        cur_x, cur_y = cur_batch
                        self.buffer.add(cur_x, cur_y)

                self._current_batch_for_buffer = None
                self._skip_buffer_addition = False
                self._current_batch_inputs = None

                if hasattr(self.strategy, "after_batch"):
                    try:
                        self.strategy.after_batch(self, x, y, loss)
                    except TypeError:
                        # keep backward compatibility if signature omits loss
                        self.strategy.after_batch(self, x, y)

        t1 = time.perf_counter()
        # Prefer strategy-specific buffers if present (e.g., Siesta latent buffer)
        buf_obj = getattr(self.strategy, "latent_buffer", self.buffer)
        try:
            buffer_size = len(buf_obj)
        except Exception:
            buffer_size = len(self.buffer)
        stats = {
            "exp_id": exp_id,
            "num_updates": num_updates,
            "train_time_sec": t1 - t0,
            "buffer_size": buffer_size,
        }
        self._last_exp_stats.append(stats)
        return stats
