"""
demo_streams.py

This demo script shows how to use the updated incremental-learning
utilities defined in clx_mvp/streams.py.

The demo includes:
  1. Building a Class-IL stream from CIFAR-10
  2. Building a Domain-IL stream using toy datasets
  3. Building a Custom stream with arbitrary datasets
  4. Visualizing per-experience class distributions

Run with:
    python demo_streams.py
"""

import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from collections import Counter

# Import your stream-building utilities
from clx_mvp.streams import (
    build_class_incremental_stream,
    build_cifar10_cil_stream,
    build_domain_incremental_stream,
    build_custom_stream,
)


# ---------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------
def visualize_experience_class_distribution(stream, title_prefix="Experience"):
    """
    Plot class distributions for each Experience in the stream.

    Args:
        stream: List[Experience]
        title_prefix: Optional prefix for plot titles.
    """
    for exp in stream:
        # Count labels inside the train dataset for this experience
        counter = Counter()

        for _, y in exp.train_ds:
            counter[int(y)] += 1

        classes = sorted(counter.keys())
        counts = [counter[c] for c in classes]

        plt.figure(figsize=(6, 4))
        plt.bar(classes, counts)
        plt.xlabel("Class ID")
        plt.ylabel("Count")
        plt.title(f"{title_prefix} {exp.exp_id}: class counts")
        plt.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------
# 1. CLASS-INCREMENTAL CIFAR-10 (generic builder)
# ---------------------------------------------------------
def demo_class_il():
    print("\n=== Demo: Generic CIFAR-10 Class-IL ===")

    transform = transforms.ToTensor()
    train = datasets.CIFAR10("data/", train=True, download=True, transform=transform)
    test  = datasets.CIFAR10("data/", train=False, download=True, transform=transform)

    stream = build_class_incremental_stream(
        train_ds=train,
        test_ds=test,
        num_classes=10,
        n_experiences=3,     # uneven class splits allowed
        label_fn=lambda ex: ex[1],
    )

    for exp in stream:
        print(
            f"Exp {exp.exp_id} | classes={exp.classes} "
            f"| train={len(exp.train_ds)} | test={len(exp.test_ds)}"
        )

    visualize_experience_class_distribution(stream, title_prefix="CIFAR-10 Class-IL")


# ---------------------------------------------------------
# 2. DOMAIN-INCREMENTAL (toy example)
# ---------------------------------------------------------
def demo_domain_il():
    print("\n=== Demo: Domain-IL (toy datasets) ===")

    # Create three toy domains with different label patterns
    def make_domain(label_values):
        x = torch.randn(len(label_values), 5)      # dummy data
        y = torch.tensor(label_values)
        return TensorDataset(x, y)

    domain_train = [
        make_domain([0, 0, 1, 1]),   # domain A
        make_domain([2, 2, 3, 3]),   # domain B
        make_domain([0, 3, 3, 0])    # domain C
    ]

    domain_test = [
        make_domain([0, 1]),
        make_domain([2, 3]),
        make_domain([0, 3]),
    ]

    stream = build_domain_incremental_stream(
        domain_train_datasets=domain_train,
        domain_test_datasets=domain_test,
        domain_names=["A", "B", "C"],
        label_fn=lambda ex: ex[1],
    )

    for exp in stream:
        print(
            f"Domain {exp.meta['domain']} | classes={exp.classes} "
            f"| train={len(exp.train_ds)} | test={len(exp.test_ds)}"
        )

    visualize_experience_class_distribution(stream, title_prefix="Domain-IL")


# ---------------------------------------------------------
# 3. CUSTOM STREAM (end-user controlled)
# ---------------------------------------------------------
def demo_custom_stream():
    print("\n=== Demo: Custom Stream ===")

    # Fake datasets
    train1 = TensorDataset(torch.randn(4, 3), torch.tensor([0, 0, 1, 1]))
    train2 = TensorDataset(torch.randn(3, 3), torch.tensor([2, 2, 2]))

    test1 = TensorDataset(torch.randn(2, 3), torch.tensor([0, 1]))
    test2 = TensorDataset(torch.randn(1, 3), torch.tensor([2]))

    stream = build_custom_stream(
        train_datasets_list=[train1, train2],
        test_datasets_list=[test1, test2],
        classes_per_exp=[[0, 1], [2]],
        meta_list=[{"type": "customA"}, {"type": "customB"}],
    )

    for exp in stream:
        print(
            f"Exp {exp.exp_id} | meta={exp.meta} | classes={exp.classes} "
            f"| train={len(exp.train_ds)} | test={len(exp.test_ds)}"
        )

    visualize_experience_class_distribution(stream, title_prefix="Custom Stream")


# ---------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    demo_class_il()
    demo_domain_il()
    demo_custom_stream()
