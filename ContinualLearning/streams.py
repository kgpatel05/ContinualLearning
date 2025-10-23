# clx_mvp/streams.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Sequence, Tuple, Optional

import random
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

# ---- Public types ----

@dataclass(frozen=True)
class Experience:
    """
    One incremental 'experience' (a.k.a. task/chunk in the stream).

    Attributes:
        exp_id: integer identifier in [0..num_experiences-1]
        train_ds: PyTorch Dataset with only this experience's training samples
        test_ds:  PyTorch Dataset with only this experience's test samples
        classes:  list of class ids included in this experience
        meta:     arbitrary dict for extra info (unused in MVP)
    """
    exp_id: int
    train_ds: Dataset
    test_ds: Dataset
    classes: List[int]
    meta: Dict[str, Any] | None = None


def build_cifar10_cil_stream(
    data_root: str,
    n_experiences: int = 5,
    seed: int = 0,
    train_transform: Optional[transforms.Compose] = None,
    test_transform: Optional[transforms.Compose] = None,
) -> List[Experience]:
    """
    Build a class-incremental CIFAR-10 stream with n_experiences (default 5),
    distributing the 10 classes uniformly (e.g., 2 per experience).

    Inputs:
        data_root: directory for torchvision datasets (downloaded if absent)
        n_experiences: number of experiences (must divide 10 for MVP)
        seed: RNG seed used for shuffling the class order
        train_transform: optional torchvision transform for train split
        test_transform: optional torchvision transform for test split

    Returns:
        List[Experience]: ordered list of experiences (exp_id ascending)

    Raises:
        ValueError: if 10 % n_experiences != 0
    """
    if 10 % n_experiences != 0:
        raise ValueError("For MVP, n_experiences must divide 10 (e.g., 5, 2, 10).")

    # default transforms
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    if test_transform is None:
        test_transform = transforms.Compose([transforms.ToTensor()])

    train = datasets.CIFAR10(data_root, train=True,  download=True, transform=train_transform)
    test  = datasets.CIFAR10(data_root, train=False, download=True, transform=test_transform)

    # partition classes → experiences
    classes = list(range(10))
    random.Random(seed).shuffle(classes)
    per = 10 // n_experiences
    buckets: List[List[int]] = [classes[i:i+per] for i in range(0, 10, per)]

    def idxs_for(ds: Dataset, allowed: Sequence[int]) -> List[int]:
        idx = []
        for i in range(len(ds)):
            # torchvision CIFAR returns (img, label)
            _, y = ds[i]
            if y in allowed:
                idx.append(i)
        return idx

    stream: List[Experience] = []
    for exp_id, cls_group in enumerate(buckets):
        tr_idx = idxs_for(train, cls_group)
        te_idx = idxs_for(test,  cls_group)
        stream.append(Experience(
            exp_id=exp_id,
            train_ds=Subset(train, tr_idx),
            test_ds=Subset(test, te_idx),
            classes=list(cls_group),
            meta=None,
        ))
    return stream
