# clx_mvp/replay.py
from __future__ import annotations
from typing import Tuple, Optional, Dict, Any, List

import random
import torch
from torch import Tensor

class ERBuffer:
    """
    Experience Replay buffer with classic reservoir sampling.
    Stores tensors on CPU.

    Public methods:
        add(x: Tensor, y: Tensor) -> None
        sample(k: int) -> Tuple[Optional[Tensor], Optional[Tensor]]
        __len__() -> int
        state_dict() -> Dict[str, Any]
        load_state_dict(state: Dict[str, Any]) -> None
    """

    def __init__(self, capacity: int) -> None:
        """
        Inputs:
            capacity: maximum number of samples retained in the buffer

        Side effects:
            Initializes empty CPU lists and a 'seen' counter.
        """
        self.capacity = int(capacity)
        self.seen: int = 0
        self._x: List[Tensor] = []
        self._y: List[Tensor] = []

    def __len__(self) -> int:
        return len(self._x)

    def add(self, x: Tensor, y: Tensor) -> None:
        """
        Admit a batch of samples via reservoir sampling.

        Inputs:
            x: [B, ...] tensor (will be stored as individual CPU tensors)
            y: [B] tensor of class ids (Long)

        Behavior:
            For each sample i in batch, increments 'seen' and either appends to
            buffer (if capacity not reached) or replaces a random index j < capacity.
        """
        assert x.size(0) == y.size(0), "x and y must have same batch size"
        B = x.size(0)
        for i in range(B):
            self._admit_one(x[i], int(y[i].item()))
            self.seen += 1

    def _admit_one(self, x_one: Tensor, y_one: int) -> None:
        x_one_cpu = x_one.detach().cpu()
        y_one_cpu = torch.tensor(y_one, dtype=torch.long)

        if len(self._x) < self.capacity:
            self._x.append(x_one_cpu)
            self._y.append(y_one_cpu)
            return

        # reservoir replacement: choose j ~ Uniform[0, seen]
        j = random.randint(0, self.seen)  # inclusive
        if j < self.capacity:
            self._x[j] = x_one_cpu
            self._y[j] = y_one_cpu

    def sample(self, k: int) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """
        Uniformly sample up to k items from the buffer.

        Inputs:
            k: number of samples requested

        Returns:
            (xs, ys):
                xs: [k, ...] tensor (on CPU) or None if buffer empty
                ys: [k] Long tensor (on CPU) or None if buffer empty
        """
        n = len(self._x)
        if n == 0:
            return None, None
        k = min(k, n)
        idx = random.sample(range(n), k)
        xs = torch.stack([self._x[i] for i in idx], dim=0)
        ys = torch.stack([self._y[i] for i in idx], dim=0).long()
        return xs, ys

    def state_dict(self) -> Dict[str, Any]:
        """
        Serialize buffer state.

        Returns:
            dict with keys: 'capacity', 'seen', 'x', 'y'
        """
        # store clones to avoid unexpected mutation
        return {
            "capacity": self.capacity,
            "seen": self.seen,
            "x": [t.clone() for t in self._x],
            "y": [t.clone() for t in self._y],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """
        Load buffer state.

        Inputs:
            state: dict returned by state_dict()

        Side effects:
            Overwrites current memory with loaded tensors.
        """
        self.capacity = int(state["capacity"])
        self.seen = int(state["seen"])
        self._x = [t.clone() for t in state["x"]]
        self._y = [t.clone() for t in state["y"]]
