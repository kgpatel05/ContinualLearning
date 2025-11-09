# clx_mvp/learner.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lightning.fabric import Fabric

from .streams import Experience
from .replay import ERBuffer
from .metrics import accuracy, average_accuracy

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
            tr_loader, te_loader = self._make_loaders(exp)
            self._train_one_experience(tr_loader)
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

    def _train_one_experience(self, train_loader: DataLoader) -> None:
        """
        One training epoch block for the current experience.

        Behavior:
            - For each batch, optionally draws replay samples and concatenates.
            - Computes CE loss, backward, step.
            - Admits only the *current* batch samples into the buffer.
        """
        self.model.train()

        for _ in range(self.epochs):
            for x, y in train_loader:
                # sample replay
                r_k = int(self.replay_ratio * x.size(0))
                rx, ry = self.buffer.sample(r_k)
                if rx is not None:
                    # move sampled CPU tensors to the same device as x
                    rx = rx.to(x.device, non_blocking=True)
                    ry = ry.to(y.device, non_blocking=True)
                    cur_x, cur_y = x, y
                    x = torch.cat([cur_x, rx], dim=0)
                    y = torch.cat([cur_y, ry], dim=0)

                # forward/backward
                logits = self.model(x)
                loss = self.criterion(logits, y)
                self.optimizer.zero_grad(set_to_none=True)
                self.fabric.backward(loss)
                self.optimizer.step()

                # admit *only* current samples (exclude replay tail if present)
                if rx is not None:
                    b_cur = cur_x.size(0)
                    self.buffer.add(cur_x.detach(), cur_y.detach())
                else:
                    self.buffer.add(x.detach(), y.detach())
