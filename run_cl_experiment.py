# experiments/run_cl_experiment.py
# python experiments/run_cl_experiment.py --algo er --dataset cifar10 --n-experiences 5

"""
Run continual learning experiments with multiple algorithms on Split CIFAR.
Supports: ER, Finetune, EWC, LwF, AGEM, BASR
"""

from __future__ import annotations
import argparse
import torch
import csv
import os
from lightning.fabric import Fabric

from clx_mvp import (
    build_cifar10_cil_stream,
    build_cifar100_cil_stream,
    build_resnet18,
    ERBuffer,
    RichERBuffer,
    Learner,
    ERStrategy,
    FinetuneStrategy,
    EWCStrategy,
    LwFStrategy,
    AGEMStrategy,
    BASRStrategy,
    ContinualEvaluator,
    accuracy,
    average_accuracy,
    compute_forgetting,
)

from clx_mvp.metrics import classification_metrics


# ---------------------------------------------------------------------
# ARGUMENTS
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CL experiment with metrics + efficiency stats")

    p.add_argument("--algo", choices=["er", "finetune", "ewc", "lwf", "agem", "basr"], default="er")
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--n-experiences", type=int, default=None)
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

    p.add_argument("--ewc-lambda", type=float, default=1000.0)
    p.add_argument("--lwf-alpha", type=float, default=1.0)
    p.add_argument("--lwf-temperature", type=float, default=2.0)
    p.add_argument("--agem-mem-size", type=int, default=256)

    return p.parse_args()


# ---------------------------------------------------------------------
# BUILDERS
# ---------------------------------------------------------------------

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


def build_strategy(args):
    if args.algo == "finetune":
        return FinetuneStrategy()

    if args.algo == "er":
        return ERStrategy(replay_ratio=args.replay_ratio)

    if args.algo == "ewc":
        base = ERStrategy(replay_ratio=args.replay_ratio) if args.replay_ratio > 0 else None
        return EWCStrategy(lambda_=args.ewc_lambda, base_strategy=base)

    if args.algo == "lwf":
        base = ERStrategy(replay_ratio=args.replay_ratio) if args.replay_ratio > 0 else None
        return LwFStrategy(alpha=args.lwf_alpha, temperature=args.lwf_temperature, base_strategy=base)

    if args.algo == "agem":
        return AGEMStrategy(mem_batch_size=args.agem_mem_size)

    if args.algo == "basr":
        return BASRStrategy(
            replay_ratio=args.replay_ratio,
            class_balance=True,
            importance_sampling=True,
        )

    raise ValueError(args.algo)


def build_buffer(args):
    if args.algo == "finetune":
        return ERBuffer(capacity=100)
    if args.algo == "basr":
        return RichERBuffer(capacity=args.buffer_capacity)
    return ERBuffer(capacity=args.buffer_capacity)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    fabric = Fabric(accelerator="auto", devices=1, precision=args.precision)
    fabric.launch()

    stream, num_classes = build_stream(args)
    model = build_resnet18(num_classes=num_classes, pretrained=False)

    buffer = build_buffer(args)
    strategy = build_strategy(args)

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

    evaluator = ContinualEvaluator(
        stream,
        fabric,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    per_exp_acc = []
    stats_log = []
    metrics_log = []

    # -----------------------------------------------------------------
    # TRAIN LOOP
    # -----------------------------------------------------------------

    for k, exp in enumerate(stream):
        learner.strategy.before_experience(learner, exp)

        tr_loader, te_loader = learner._make_loaders(exp)
        stats = learner._train_one_experience(tr_loader, exp_id=exp.exp_id)

        learner.strategy.after_experience(learner, exp)

        acc = accuracy(learner.model, te_loader, fabric)
        per_exp_acc.append(acc)

        evaluator.evaluate_after_exp(learner.model, upto_exp=k)

        metrics = classification_metrics(
            learner.model,
            te_loader,
            fabric,
            num_classes=num_classes,
        )

        precision = metrics.get("precision", 0.0)
        recall = metrics.get("recall", 0.0)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        metrics_log.append({
            "exp_id": exp.exp_id,
            "accuracy": metrics.get("accuracy", acc),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

        stats_log.append(stats)

        fabric.print(
            f"[Exp {exp.exp_id}] "
            f"acc={acc:.2f}% "
            f"AA={average_accuracy(per_exp_acc):.2f}% "
            f"updates={stats['num_updates']} "
            f"time={stats['train_time_sec']:.1f}s "
            f"buffer={stats['buffer_size']}"
        )

    # -----------------------------------------------------------------
    # FINAL STATS
    # -----------------------------------------------------------------

    forgetting = compute_forgetting(evaluator.acc_matrix)
    avg_forgetting = sum(forgetting) / len(forgetting) if forgetting else 0.0

    fabric.print("\nFINAL RESULTS")
    fabric.print(f"Average accuracy: {average_accuracy(per_exp_acc):.2f}%")
    fabric.print(f"Average forgetting: {avg_forgetting:.2f}%")

    # -----------------------------------------------------------------
    # SAVE CSV (CORRECT)
    # -----------------------------------------------------------------

    if fabric.global_rank == 0:
        csv_name = f"metrics_seed{args.seed}.csv"

        with open(csv_name, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "exp_id",
                    "accuracy",
                    "average_accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "num_updates",
                    "train_time_sec",
                    "buffer_size",
                ],
            )
            writer.writeheader()

            running_acc = []
            for i in range(len(metrics_log)):
                running_acc.append(metrics_log[i]["accuracy"])

                writer.writerow({
                    "exp_id": metrics_log[i]["exp_id"],
                    "accuracy": metrics_log[i]["accuracy"],
                    "average_accuracy": sum(running_acc) / len(running_acc),
                    "precision": metrics_log[i]["precision"],
                    "recall": metrics_log[i]["recall"],
                    "f1": metrics_log[i]["f1"],
                    "num_updates": stats_log[i]["num_updates"],
                    "train_time_sec": stats_log[i]["train_time_sec"],
                    "buffer_size": stats_log[i]["buffer_size"],
                })

        fabric.print(f"\nMetrics saved to {csv_name}")


if __name__ == "__main__":
    main()
