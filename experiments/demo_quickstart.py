"""
Quickstart driver for the continual learning toolkit.

This script is designed to be easy to read and tweak. It builds a tiny synthetic
class-incremental stream, constructs a small model, and lets you pick a strategy
at the command line (ER, Finetune, GRASP, SIESTA, SGM wrapper).

Example runs (fast on CPU):
    python experiments/demo_quickstart.py --algo er
    python experiments/demo_quickstart.py --algo grasp --replay-ratio 0.5
    python experiments/demo_quickstart.py --algo siesta --sleep-every 2
    python experiments/demo_quickstart.py --algo sgm --base er
"""
from __future__ import annotations
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from lightning.fabric import Fabric

from clx_mvp import (
    Experience,
    Learner,
    ERBuffer,
    RichERBuffer,
    ERStrategy,
    FinetuneStrategy,
)
from clx_mvp.strategies import (
    GraspStrategy,
    GraspConfig,
    SiestaStrategy,
    SiestaConfig,
    SleepScheduleConfig,
    SgmStrategy,
    SgmConfig,
)


# ---- minimal helpers ----

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class TinyMLP(nn.Module):
    """Small, fast network for demos."""

    def __init__(self, in_dim: int = 20, hidden: int = 64, num_classes: int = 3):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
        )
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature(x)
        return self.fc(h)


def build_synthetic_stream(num_classes: int, samples_per_class: int, in_dim: int = 20):
    """Returns a one-experience stream for speed; tweak to add more experiences."""
    xs, ys = [], []
    for cls in range(num_classes):
        for _ in range(samples_per_class):
            xs.append(torch.randn(in_dim))
            ys.append(torch.tensor(cls))
    ds = TensorDataset(torch.stack(xs), torch.stack(ys))
    exp = Experience(exp_id=0, train_ds=ds, test_ds=ds, classes=list(range(num_classes)))
    return [exp], num_classes


# ---- strategy/buffer builders ----

def build_buffer(algo: str, capacity: int) -> torch.nn.Module:
    if algo == "grasp":
        return RichERBuffer(capacity=capacity)
    return ERBuffer(capacity=capacity)


def build_strategy(args, num_classes: int):
    if args.algo == "finetune":
        return FinetuneStrategy()

    if args.algo == "er":
        return ERStrategy(replay_ratio=args.replay_ratio)

    if args.algo == "grasp":
        grasp_cfg = GraspConfig(replay_ratio=args.replay_ratio)
        return GraspStrategy(grasp_cfg)

    if args.algo == "siesta":
        siesta_cfg = SiestaConfig(
            buffer_size=args.latent_buffer,
            sleep_steps=args.sleep_steps,
            sleep_batch_size=args.batch_size,
            schedule=SleepScheduleConfig(
                sleep_every_batches=args.sleep_every,
                sleep_on_task_end=True,
                min_buffer_fraction=0.0,
            ),
        )
        return SiestaStrategy(siesta_cfg)

    if args.algo == "sgm":
        base = ERStrategy(replay_ratio=args.replay_ratio) if args.base == "er" else FinetuneStrategy()
        sgm_cfg = SgmConfig(use_lora=args.use_lora, use_oocf=args.use_oocf)
        return SgmStrategy(sgm_cfg, base_strategy=base)

    raise ValueError(f"Unknown algo: {args.algo}")


# ---- CLI ----

def parse_args():
    p = argparse.ArgumentParser(description="Small, self-contained continual learning demo.")
    p.add_argument("--algo", choices=["er", "finetune", "grasp", "siesta", "sgm"], default="er")
    p.add_argument("--base", choices=["er", "finetune"], default="er", help="Base strategy for SGM wrapper.")
    p.add_argument("--replay-ratio", type=float, default=0.5, help="Replay mix ratio for ER/GRASP/SGM base.")
    p.add_argument("--buffer-capacity", type=int, default=200, help="Replay buffer capacity for raw samples.")
    p.add_argument("--latent-buffer", type=int, default=400, help="Latent buffer capacity for SIESTA sleep phase.")
    p.add_argument("--sleep-every", type=int, default=2, help="Wake batches between SIESTA sleep phases.")
    p.add_argument("--sleep-steps", type=int, default=4, help="Optimization steps per SIESTA sleep phase.")
    p.add_argument("--use-lora", action="store_true", help="Enable LoRA inside SGM.")
    p.add_argument("--use-oocf", action="store_true", help="Enable output freezing inside SGM.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--samples-per-class", type=int, default=40, help="Size of the synthetic dataset.")
    p.add_argument("--num-classes", type=int, default=3)
    return p.parse_args()


# ---- main ----

def main():
    args = parse_args()
    set_seed(args.seed)

    stream, num_classes = build_synthetic_stream(
        num_classes=args.num_classes,
        samples_per_class=args.samples_per_class,
        in_dim=20,
    )

    model = TinyMLP(in_dim=20, hidden=64, num_classes=num_classes)
    buffer = build_buffer(args.algo, args.buffer_capacity)
    strategy = build_strategy(args, num_classes)

    fabric = Fabric(accelerator="cpu", devices=1, precision="32-true")
    fabric.launch()

    learner = Learner(
        model=model,
        fabric=fabric,
        buffer=buffer,
        strategy=strategy,
        batch_size=args.batch_size,
        epochs=args.epochs,
        replay_ratio=args.replay_ratio,
        num_workers=0,
        pin_memory=False,
    )

    report = learner.fit(stream)
    fabric.print(f"\nCompleted demo with algo={args.algo}")
    fabric.print(f"Per-exp acc: {report.per_exp_acc}")
    fabric.print(f"Average acc: {report.avg_acc:.2f}%")
    if hasattr(strategy, "sleep_log"):
        fabric.print(f"SIESTA sleep events: {len(strategy.sleep_log)}")
    if hasattr(strategy, "policy"):
        fabric.print(f"GRASP sampled difficulties (last batch): {getattr(strategy.policy, 'last_sampled_scores', [])}")


if __name__ == "__main__":
    main()
