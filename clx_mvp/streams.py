# clx_mvp/streams.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Sequence, Tuple, Optional, Callable

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


def build_class_incremental_stream(
    train_ds,
    test_ds,
    num_classes: int,
    n_experiences: int,
    class_order: Optional[list[int]] = None,
    label_fn: Callable = lambda ex: ex[1],
) -> list[Experience]:
    """
    Generic builder for class-incremental (Class-IL) streams.

    The label space {0, ..., num_classes-1} is partitioned into
    `n_experiences` chunks (the last chunk may be uneven), and for each
    chunk an `Experience` is created containing only those classes.

    Args:
        train_ds: Training dataset (e.g., CIFAR-10 train split) whose labels
            are obtained via `label_fn`.
        test_ds: Test dataset, with the same label space and `label_fn` as
            `train_ds`.
        num_classes: Total number of distinct classes in the label space.
        n_experiences: Desired number of incremental experiences.
        class_order: Optional permutation of [0, ..., num_classes-1] that
            fixes the order in which classes appear across experiences. If
            None, the natural order is used.
        label_fn: Function that extracts a label from a dataset example
            (e.g., lambda ex: ex[1] for (x, y) tuples).

    Returns:
        A list of `Experience` objects in chronological order (exp_id
        increasing), each representing a chunk of classes.
    """
    if class_order is None:
        class_order = list(range(num_classes))

    assert len(class_order) == num_classes

    # Compute how many classes per experience to ensure that we cover all classes even if num_classes is not divisible
    # by n_experiences; the last exp may have fewer classes.
    chunk = (num_classes + n_experiences - 1) // n_experiences

    experiences: List[Experience] = []

    for k in range(n_experiences):
        cls = class_order[k * chunk:(k+1)*chunk]

        if not cls:
            break

        train_ind = get_class_indices(train_ds, cls, label_fn)
        test_ind = get_class_indices(test_ds, cls, label_fn)

        train_subset = Subset(train_ds, train_ind)
        test_subset = Subset(test_ds, test_ind)

        exp = Experience(
            exp_id = k,
            train_ds = train_subset,
            test_ds = test_subset,
            classes = cls,
            meta = {"type": "class-il"},
        )

        experiences.append(exp)

    return experiences


def build_domain_incremental_stream(
    domain_train_datasets: list[Dataset],
    domain_test_datasets: list[Dataset],
    domain_names: Optional[list[str]] = None,
    label_fn: Callable = lambda ex: ex[1],
) -> list[Experience]:
    """
    Build a domain-incremental (Domain-IL) stream, where each experience
    corresponds to a separate domain (dataset pair) but shares (conceptually)
    the same label space.

    Each domain gets its own Experience, and for each one we also discover
    which classes actually appear in its training set.
    """
    # Each train dataset must have a corresponding test dataset.
    assert len(domain_train_datasets) == len(domain_test_datasets), \
        "domain_train_datasets and domain_test_datasets must have same length."

    # If domain_names are provided, they must match the number of domains.
    if domain_names is not None:
        assert len(domain_names) == len(domain_train_datasets), \
            "domain_names must match number of domains."

    experiences: list[Experience] = []

    for k, (tr, te) in enumerate(zip(domain_train_datasets, domain_test_datasets)):
        # Discover which classes appear in this domain by scanning the train set.
        seen_classes: set[int] = set()
        for i in range(len(tr)):
            y = label_fn(tr[i])
            seen_classes.add(int(y))

        # Stable, sorted list of classes for this domain.
        classes_in_domain = sorted(seen_classes)

        # Choose a domain name, either user-provided or a fallback.
        if domain_names is not None:
            domain_name = domain_names[k]
        else:
            domain_name = f"domain_{k}"

        exp = Experience(
            exp_id=k,
            train_ds=tr,
            test_ds=te,
            classes=classes_in_domain,
            meta={
                "type": "domain-il",
                "domain": domain_name,
            },
        )
        experiences.append(exp)

    return experiences

def build_custom_stream(
    train_datasets_list,
    test_datasets_list,
    classes_per_exp,
    mata_list=None,
) -> List[Experience]:
    """
    Build a custom stream where the user provides per-experience datasets,
    classes, and (optionally) meta information.

    Args:
        train_datasets_list: Sequence of training datasets, one per experience.
        test_datasets_list: Sequence of test datasets, aligned with
            train_datasets_list by position.
        classes_per_exp: Sequence where each element is the list (or iterable)
            of class IDs associated with that experience.
        mata_list: Optional sequence of meta dicts (or None). If provided,
            mata_list[k] is stored as the `meta` field of experience k.
            If None, a default meta dict is used.

    Returns:
        List[Experience]: one Experience per (train, test, classes) triple.
    """
    n_exps = len(train_datasets_list)

    # Basic consistency checks.
    assert len(test_datasets_list) == n_exps, \
        "test_datasets_list must have same length as train_datasets_list."
    assert len(classes_per_exp) == n_exps, \
        "classes_per_exp must have same length as train_datasets_list."

    if mata_list is not None:
        assert len(mata_list) == n_exps, \
            "mata_list must have same length as train_datasets_list."

    experiences: List[Experience] = []

    for k in range(n_exps):
        train_ds = train_datasets_list[k]
        test_ds = test_datasets_list[k]
        cls = list(classes_per_exp[k])

        if mata_list is not None:
            meta = mata_list[k]
        else:
            meta = {"type": "custom", "exp_id": k}

        exp = Experience(
            exp_id=k,
            train_ds=train_ds,
            test_ds=test_ds,
            classes=cls,
            meta=meta,
        )
        experiences.append(exp)

    return experiences


def split_dataset_by_indices(
    dataset,
    indices_per_split
) -> list[Subset]:
    """
    Create a list of `Subset` views on `dataset`, one per list of indices.

    Args:
        dataset: Base dataset to be split.
        indices_per_split: Iterable of index collections, where each element
            is a sequence (e.g., list) of integer indices into `dataset`.

    Returns:
        A list of `torch.utils.data.Subset` objects, one for each
        index sequence in `indices_per_split`.
    """
    subsets = []

    for idxs in indices_per_split:
        subset = Subset(dataset, idxs)
        subsets.append(subset)
    
    return subsets


def get_class_indices(
    dataset,
    target_classes,
    label_fn: Callable
) -> list[int]:
    """
    Return the indices of all examples in `dataset` whose label is in
    `target_classes`.

    Args:
        dataset: Dataset providing examples, typically returning (x, y) or
            some structure that `label_fn` can consume.
        target_classes: Iterable of integer class ids to keep.
        label_fn: Callable that, given a dataset example (e.g., dataset[i]),
            returns its label as an int (or something convertible to int).

    Returns:
        List of integer indices i such that label_fn(dataset[i]) is one of
        the target_classes.
    """
    matching_indices: list[int] = []
    target_set = set(int(c) for c in target_classes)

    for i in range(len(dataset)):
        label = label_fn(dataset[i])
        label = int(label)
        if label in target_set:
            matching_indices.append(i)

    return matching_indices