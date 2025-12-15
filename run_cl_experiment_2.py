# run_cl_experiment_2.py
# python run_cl_experiment_2.py --algo ewc

from __future__ import annotations
import argparse
import torch
import csv
import os
from torch.utils.data import DataLoader, ConcatDataset
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
)
from clx_mvp import accuracy, average_accuracy, compute_forgetting

from sklearn.metrics import precision_score, recall_score, f1_score

# -----------------------------
# ARGUMENTS
# -----------------------------
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


# -----------------------------
# STREAM & STRATEGY BUILDERS
# -----------------------------
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
        return BASRStrategy(replay_ratio=args.replay_ratio, class_balance=True, importance_sampling=True)
    raise ValueError(args.algo)

def build_buffer(args):
    if args.algo == "finetune":
        return ERBuffer(capacity=100)
    if args.algo == "basr":
        return RichERBuffer(capacity=args.buffer_capacity)
    return ERBuffer(capacity=args.buffer_capacity)


# -----------------------------
# METRICS FUNCTIONS
# -----------------------------
def compute_masked_accuracy(model, dataloader, allowed_classes, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb, *_ in dataloader:
            mask = torch.tensor([y.item() in allowed_classes for y in yb])
            if not mask.any():
                continue
            xb = xb[mask].to(device)
            yb = yb[mask].to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total if total > 0 else 0.0

def compute_prf(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb, *_ in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return precision, recall, f1


# -----------------------------
# MAIN
# -----------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Force FP32 to avoid hanging
    fabric = Fabric(accelerator="auto", devices=1, precision="32")
    fabric.launch()
    fabric.print("Fabric launched successfully.")
    fabric.print(f"Device: {fabric.device}")

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

    seen_classes = set()
    metrics_log = []

    # TRAIN LOOP
    for exp_id, exp in enumerate(stream):
        # Updated attribute from 'classes_in_this_experience' to 'classes'
        exp_classes = getattr(exp, "classes", None)
        if exp_classes is None:
            raise AttributeError(f"Experience {exp_id} has no attribute 'classes'")

        fabric.print(f"\n--- Starting Experience {exp.exp_id}, Classes: {exp_classes} ---")

        learner.strategy.before_experience(learner, exp)
        tr_loader, te_loader = learner._make_loaders(exp)
        stats = learner._train_one_experience(tr_loader, exp_id=exp.exp_id)
        learner.strategy.after_experience(learner, exp)

        # Update seen classes
        seen_classes.update(exp_classes)
        current_classes = set(exp_classes)

        # Build seen-only test loader
        seen_test_dataset = ConcatDataset([stream[i].train_ds for i in range(exp_id + 1)])
        test_loader = DataLoader(seen_test_dataset, batch_size=args.batch_size, shuffle=False)


        # Compute masked metrics
        exp_acc = compute_masked_accuracy(learner.model, test_loader, current_classes, device=fabric.device)
        avg_acc = compute_masked_accuracy(learner.model, test_loader, seen_classes, device=fabric.device)
        precision, recall, f1 = compute_prf(learner.model, test_loader, device=fabric.device)

        metrics_log.append({
            "exp_id": exp.exp_id,
            "experience_accuracy": exp_acc,
            "average_accuracy": avg_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "num_updates": stats["num_updates"],
            "train_time_sec": stats["train_time_sec"],
            "buffer_size": stats["buffer_size"],
        })

        fabric.print(
            f"[Exp {exp.exp_id}] "
            f"Experience Acc={exp_acc:.4f}, Avg Acc={avg_acc:.4f}, "
            f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, "
            f"Updates={stats['num_updates']}, Time={stats['train_time_sec']:.1f}s, Buffer={stats['buffer_size']}"
        )

    # SAVE CSV
    if fabric.global_rank == 0:
        os.makedirs("figures", exist_ok=True)
        csv_file = f"metrics_{args.algo}.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "exp_id","experience_accuracy","average_accuracy","precision","recall","f1",
                "num_updates","train_time_sec","buffer_size"
            ])
            writer.writeheader()
            for row in metrics_log:
                writer.writerow(row)
        fabric.print(f"\nMetrics saved to {csv_file}")


if __name__ == "__main__":
    main()
