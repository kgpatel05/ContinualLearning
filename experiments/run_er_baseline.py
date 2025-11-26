# experiments/run_er_baseline.py
# python experiments/run_er_baseline.py --dataset cifar100 --data-root ./data --n-experiences 10
# python experiments/run_er_baseline.py --dataset cifar10 --data-root ./data --n-experiences 5

"""
Run a simple ER baseline with accuracy matrix + efficiency stats on Split CIFAR.
"""

from __future__ import annotations
import argparse
import torch
from lightning.fabric import Fabric

from clx_mvp import (
    build_cifar10_cil_stream,
    build_cifar100_cil_stream,
    build_resnet18,
    ERBuffer,
    Learner,
    ERStrategy,
    ContinualEvaluator,
    accuracy,
    average_accuracy,
    compute_forgetting,
)


def parse_args():
    p = argparse.ArgumentParser(description="ER baseline with accuracy matrix + efficiency stats")
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--n-experiences", type=int, default=None, help="If unset: 5 for CIFAR-10, 10 for CIFAR-100")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--buffer-capacity", type=int, default=2000)
    p.add_argument("--replay-ratio", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    return p.parse_args()


def build_stream(args):
    if args.dataset == "cifar10":
        n_exp = args.n_experiences or 5
        stream = build_cifar10_cil_stream(args.data_root, n_experiences=n_exp, seed=args.seed)
        num_classes = 10
    else:
        n_exp = args.n_experiences or 10
        stream = build_cifar100_cil_stream(args.data_root, n_experiences=n_exp, seed=args.seed)
        num_classes = 100
    return stream, num_classes


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    fabric = Fabric(accelerator="auto", devices=1, precision=args.precision)
    fabric.launch()

    stream, num_classes = build_stream(args)
    model = build_resnet18(num_classes=num_classes, pretrained=False)
    buffer = ERBuffer(capacity=args.buffer_capacity)
    strategy = ERStrategy(replay_ratio=args.replay_ratio)

    learner = Learner(
        model=model,
        fabric=fabric,
        buffer=buffer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        replay_ratio=args.replay_ratio,
        strategy=strategy,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    evaluator = ContinualEvaluator(stream, fabric, batch_size=args.batch_size, num_workers=args.num_workers)

    per_exp_acc = []
    acc_matrix = []
    stats_log = []

    for k, exp in enumerate(stream):
        learner.strategy.before_experience(learner, exp)
        tr_loader, te_loader = learner._make_loaders(exp)
        stats = learner._train_one_experience(tr_loader, exp_id=exp.exp_id)
        learner.strategy.after_experience(learner, exp)

        acc = accuracy(learner.model, te_loader, fabric)
        per_exp_acc.append(acc)
        acc_row = evaluator.evaluate_after_exp(learner.model, upto_exp=k)
        acc_matrix.append(acc_row)
        stats_log.append(stats)

        fabric.print(
            f"[Exp {exp.exp_id}] classes={exp.classes} acc={acc:.2f}% "
            f"AA={average_accuracy(per_exp_acc):.2f}% "
            f"updates={stats['num_updates']} time={stats['train_time_sec']:.1f}s "
            f"buffer={stats['buffer_size']}"
        )

    forgetting = compute_forgetting(evaluator.acc_matrix)
    fabric.print("\nAccuracy matrix:")
    for row in evaluator.acc_matrix:
        fabric.print(row)
    fabric.print(f"Forgetting: {forgetting}")
    fabric.print(f"Per-exp acc: {per_exp_acc}")
    fabric.print(f"Average acc: {average_accuracy(per_exp_acc):.2f}%")

    if fabric.global_rank == 0:
        fabric.print(f"Efficiency stats: {stats_log}")


if __name__ == "__main__":
    main()
