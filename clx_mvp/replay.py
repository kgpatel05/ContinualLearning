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


class RichERBuffer:
    """
    Extended buffer that tracks importance scores, class ids, and age.
    Provides importance-aware add and class-balanced sampling.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._entries: List[Dict[str, Any]] = []
        self._age_counter: int = 0

    def __len__(self) -> int:
        return len(self._entries)

    def add(
        self,
        x: Tensor,
        y: Tensor,
        scores: Optional[Tensor] = None,
        features: Optional[Tensor] = None,
    ) -> None:
        """
        Add a batch with optional per-sample importance scores.
        If buffer is full, replaces the lowest-score entries first.
        """
        assert x.size(0) == y.size(0), "x and y must have same batch size"
        B = x.size(0)
        if scores is None:
            scores = torch.ones(B, device=x.device)
        if scores.numel() != B:
            raise ValueError("scores must match batch size")

        for i in range(B):
            self._admit_one(
                x[i].detach().cpu(),
                int(y[i].item()),
                float(scores[i].detach().cpu().item()),
                feat=features[i].detach().cpu() if features is not None else None,
            )

    def _admit_one(self, x_one: Tensor, y_one: int, score: float, feat: Optional[Tensor] = None) -> None:
        entry = {
            "x": x_one,
            "y": torch.tensor(y_one, dtype=torch.long),
            "score": float(score),
            "class": int(y_one),
            "age": self._age_counter,
            "feat": feat,
        }
        self._age_counter += 1

        if len(self._entries) < self.capacity:
            self._entries.append(entry)
            return

        # importance-aware replacement: kick out lowest-score entry if incoming is higher
        min_idx, min_entry = min(enumerate(self._entries), key=lambda kv: kv[1]["score"])
        if score > min_entry["score"]:
            self._entries[min_idx] = entry

    def sample(
        self,
        k: int,
        *,
        class_balance: bool = True,
        importance_sampling: bool = True,
        return_features: bool = False,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]] | Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """
        Sample up to k items, optionally class-balanced and/or importance-weighted.
        """
        n = len(self._entries)
        if n == 0:
            return None, None
        k = min(k, n)

        if class_balance:
            idx = self._class_balanced_indices(k)
        elif importance_sampling:
            idx = self._importance_indices(k)
        else:
            idx = random.sample(range(n), k)

        xs = torch.stack([self._entries[i]["x"] for i in idx], dim=0)
        ys = torch.stack([self._entries[i]["y"] for i in idx], dim=0).long()
        if return_features:
            feats = [self._entries[i].get("feat") for i in idx]
            if all(f is not None for f in feats):
                feat_tensor = torch.stack([f for f in feats], dim=0)  # type: ignore[arg-type]
            else:
                feat_tensor = None
            return xs, ys, feat_tensor
        return xs, ys

    def _class_balanced_indices(self, k: int) -> List[int]:
        by_cls: Dict[int, List[int]] = {}
        for idx, e in enumerate(self._entries):
            by_cls.setdefault(e["class"], []).append(idx)

        selected: List[int] = []
        classes = list(by_cls.keys())
        cls_ptr = 0
        while len(selected) < k:
            cls = classes[cls_ptr % len(classes)]
            pool = by_cls[cls]
            chosen = random.choice(pool)
            selected.append(chosen)
            cls_ptr += 1
        return selected

    def _importance_indices(self, k: int) -> List[int]:
        scores = torch.tensor([e["score"] for e in self._entries], dtype=torch.float)
        probs = scores / (scores.sum() + 1e-12)
        chosen = torch.multinomial(probs, k, replacement=False).tolist()
        return chosen

    def state_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "age": self._age_counter,
            "entries": [
                {
                    "x": e["x"].clone(),
                    "y": e["y"].clone(),
                    "score": e["score"],
                    "class": e["class"],
                    "age": e["age"],
                    "feat": e.get("feat"),
                }
                for e in self._entries
            ],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.capacity = int(state["capacity"])
        self._age_counter = int(state.get("age", 0))
        self._entries = [
            {
                "x": ent["x"].clone(),
                "y": ent["y"].clone(),
                "score": float(ent["score"]),
                "class": int(ent["class"]),
                "age": int(ent.get("age", 0)),
                "feat": ent.get("feat"),
            }
            for ent in state["entries"]
        ]


class LatentReplayBuffer:
    """
    Reservoir-sampled buffer for compressed latent representations.
    Stores encoded latents plus optional auxiliary data (e.g., scales) and metadata.
    """
    def __init__(self, capacity: int, replacement: str = "reservoir") -> None:
        self.capacity = int(capacity)
        self.replacement = replacement
        self._entries: List[Dict[str, Any]] = []
        self.seen: int = 0

    def __len__(self) -> int:
        return len(self._entries)

    def add(
        self,
        encoded_latents: Tensor,
        labels: Tensor,
        *,
        aux: Optional[Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a batch of encoded latents.

        Args:
            encoded_latents: compressed latents (typically CPU).
            labels: class labels.
            aux: optional per-sample aux data (e.g., quantization scales) aligned with batch.
            metadata: optional metadata dict attached to every sample in the batch.
        """
        assert encoded_latents.size(0) == labels.size(0), "latent/label batch mismatch"
        B = encoded_latents.size(0)
        aux_tensor = None
        if aux is not None:
            if aux.dim() == 0 or aux.size(0) != B:
                aux_tensor = aux.detach().cpu().expand(B, *([1] * (encoded_latents.dim() - 1)))
            else:
                aux_tensor = aux.detach().cpu()

        for i in range(B):
            entry_meta = {} if metadata is None else dict(metadata)
            self._admit_one(
                encoded_latents[i].detach().cpu(),
                int(labels[i].item()),
                aux_tensor[i].detach().cpu() if aux_tensor is not None else None,
                entry_meta,
            )
            self.seen += 1

    def _admit_one(
        self,
        latent_one: Tensor,
        label_one: int,
        aux_one: Optional[Tensor],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        entry = {
            "latent": latent_one,
            "label": torch.tensor(label_one, dtype=torch.long),
            "aux": aux_one,
            "meta": metadata or {},
        }

        if len(self._entries) < self.capacity:
            self._entries.append(entry)
            return

        if self.replacement == "fifo":
            self._entries.pop(0)
            self._entries.append(entry)
            return

        # default: reservoir
        j = random.randint(0, self.seen)
        if j < self.capacity:
            self._entries[j] = entry

    def sample(
        self,
        k: int,
        *,
        return_metadata: bool = False,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]] | Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[List[Dict[str, Any]]]]:
        """
        Sample compressed latents.

        Returns:
            encoded_latents, labels, aux [, metadata]
        """
        n = len(self._entries)
        if n == 0:
            if return_metadata:
                return None, None, None, None
            return None, None, None
        k = min(k, n)
        idx = random.sample(range(n), k)
        latents = torch.stack([self._entries[i]["latent"] for i in idx], dim=0)
        labels = torch.stack([self._entries[i]["label"] for i in idx], dim=0).long()
        aux = None
        aux_values = [self._entries[i]["aux"] for i in idx]
        if any(a is not None for a in aux_values):
            template = next((a for a in aux_values if a is not None), None)
            if template is None:
                template = torch.zeros_like(latents)
            aux = torch.stack(
                [
                    a if a is not None else torch.zeros_like(template)  # type: ignore[arg-type]
                    for a in aux_values
                ],
                dim=0,
            )
        if return_metadata:
            metas = [self._entries[i].get("meta", {}) for i in idx]
            return latents, labels, aux, metas
        return latents, labels, aux

    def state_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "replacement": self.replacement,
            "seen": self.seen,
            "entries": [
                {
                    "latent": e["latent"].clone(),
                    "label": e["label"].clone(),
                    "aux": None if e["aux"] is None else e["aux"].clone(),
                    "meta": dict(e.get("meta", {})),
                }
                for e in self._entries
            ],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.capacity = int(state["capacity"])
        self.replacement = state.get("replacement", "reservoir")
        self.seen = int(state.get("seen", 0))
        self._entries = [
            {
                "latent": ent["latent"].clone(),
                "label": ent["label"].clone(),
                "aux": None if ent.get("aux") is None else ent["aux"].clone(),
                "meta": dict(ent.get("meta", {})),
            }
            for ent in state.get("entries", [])
        ]
