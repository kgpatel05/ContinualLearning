# experiments/run_cl_experiment.py
# python experiments/run_cl_experiment.py --algo er --dataset cifar100 --n-experiences 10
# python experiments/run_cl_experiment.py --algo finetune --dataset cifar10 --n-experiences 5
# python experiments/run_cl_experiment.py --algo ewc --dataset cifar10 --n-experiences 5

"""
Run continual learning experiments with multiple algorithms on Split CIFAR.
Supports: ER, Finetune, EWC, LwF, AGEM, BASR
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


def parse_args():
    p = argparse.ArgumentParser(description="CL experiment with accuracy matrix + efficiency stats")
    # Algorithm selection
    p.add_argument(
        "--algo",
        choices=["er", "finetune", "ewc", "lwf", "agem", "basr"],
        default="er",
        help="Continual learning algorithm to use"
    )
    
    # Dataset configuration
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--n-experiences", type=int, default=None, help="If unset: 5 for CIFAR-10, 10 for CIFAR-100")
    p.add_argument("--seed", type=int, default=0)
    
    # Buffer configuration
    p.add_argument("--buffer-capacity", type=int, default=2000)
    p.add_argument("--replay-ratio", type=float, default=0.5, help="For replay-based methods")
    
    # Training configuration
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    
    # Algorithm-specific hyperparameters
    p.add_argument("--ewc-lambda", type=float, default=1000.0, help="EWC regularization strength")
    p.add_argument("--lwf-alpha", type=float, default=1.0, help="LwF distillation weight")
    p.add_argument("--lwf-temperature", type=float, default=2.0, help="LwF distillation temperature")
    p.add_argument("--agem-mem-size", type=int, default=256, help="AGEM reference batch size")
    
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


def build_strategy(args):
    """Build the CL strategy based on algorithm selection."""
    if args.algo == "finetune":
        return FinetuneStrategy()
    
    elif args.algo == "er":
        return ERStrategy(replay_ratio=args.replay_ratio)
    
    elif args.algo == "ewc":
        # EWC can optionally wrap ER for rehearsal + regularization
        base = ERStrategy(replay_ratio=args.replay_ratio) if args.replay_ratio > 0 else None
        return EWCStrategy(lambda_=args.ewc_lambda, base_strategy=base)
    
    elif args.algo == "lwf":
        # LwF can optionally wrap ER
        base = ERStrategy(replay_ratio=args.replay_ratio) if args.replay_ratio > 0 else None
        return LwFStrategy(alpha=args.lwf_alpha, temperature=args.lwf_temperature, base_strategy=base)
    
    elif args.algo == "agem":
        return AGEMStrategy(mem_batch_size=args.agem_mem_size)
    
    elif args.algo == "basr":
        return BASRStrategy(
            replay_ratio=args.replay_ratio,
            class_balance=True,
            importance_sampling=True
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")


def build_buffer(args):
    """Build the appropriate buffer for the selected algorithm."""
    if args.algo == "finetune":
        # Finetune doesn't use buffer, but Learner requires one
        return ERBuffer(capacity=100)
    
    elif args.algo == "basr":
        # BASR requires RichERBuffer for importance tracking
        return RichERBuffer(capacity=args.buffer_capacity)
    
    else:
        # All other methods use standard ERBuffer
        return ERBuffer(capacity=args.buffer_capacity)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    fabric = Fabric(accelerator="auto", devices=1, precision=args.precision)
    fabric.launch()

    stream, num_classes = build_stream(args)
    model = build_resnet18(num_classes=num_classes, pretrained=False)
    buffer = build_buffer(args)
    strategy = build_strategy(args)

    fabric.print(f"\n{'='*60}")
    fabric.print(f"Algorithm: {args.algo.upper()}")
    fabric.print(f"Dataset: {args.dataset.upper()} ({len(stream)} experiences)")
    fabric.print(f"Buffer capacity: {args.buffer_capacity}")
    fabric.print(f"Replay ratio: {args.replay_ratio}")
    fabric.print(f"{'='*60}\n")

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
    stats_log = []

    for k, exp in enumerate(stream):
        learner.strategy.before_experience(learner, exp)
        tr_loader, te_loader = learner._make_loaders(exp)
        stats = learner._train_one_experience(tr_loader, exp_id=exp.exp_id)
        learner.strategy.after_experience(learner, exp)

        acc = accuracy(learner.model, te_loader, fabric)
        per_exp_acc.append(acc)
        acc_row = evaluator.evaluate_after_exp(learner.model, upto_exp=k)
        stats_log.append(stats)

        fabric.print(
            f"[Exp {exp.exp_id}] classes={exp.classes} acc={acc:.2f}% "
            f"AA={average_accuracy(per_exp_acc):.2f}% "
            f"updates={stats['num_updates']} time={stats['train_time_sec']:.1f}s "
            f"buffer={stats['buffer_size']}"
        )

    forgetting = compute_forgetting(evaluator.acc_matrix)
    avg_forgetting = sum(forgetting) / len(forgetting) if forgetting else 0.0
    total_time = sum(s['train_time_sec'] for s in stats_log)
    total_updates = sum(s['num_updates'] for s in stats_log)

    fabric.print(f"\n{'='*60}")
    fabric.print("FINAL RESULTS")
    fabric.print(f"{'='*60}")
    fabric.print(f"Average accuracy: {average_accuracy(per_exp_acc):.2f}%")
    fabric.print(f"Average forgetting: {avg_forgetting:.2f}%")
    fabric.print(f"Total training time: {total_time:.1f}s")
    fabric.print(f"Total updates: {total_updates}")
    fabric.print(f"\nPer-task forgetting: {[f'{f:.2f}' for f in forgetting]}")
    
    if fabric.global_rank == 0:
        fabric.print(f"\nAccuracy matrix:")
        for i, row in enumerate(evaluator.acc_matrix):
            fabric.print(f"  After exp {i}: {[f'{a:.2f}' for a in row]}")


if __name__ == "__main__":
    main()
