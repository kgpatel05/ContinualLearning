# experiments/run_cl_experiment.py
# python experiments/run_cl_experiment.py --algo er --dataset cifar100 --n-experiences 10 --backend clx
# python experiments/run_cl_experiment.py --algo ewc --dataset cifar10 --n-experiences 5 --backend avalanche

# python experiments/plot_results.py --inputs "experiments/results/*.json" --outdir experiments/plots

"""
Run continual learning experiments across clx_mvp and Avalanche backends.
Logs per-experience/per-class metrics, confusion matrices, and writes JSON for plotting.
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from lightning.fabric import Fabric
from torchvision import transforms

from clx_mvp import (
    AGEMStrategy,
    BASRStrategy,
    ContinualEvaluator,
    ERBuffer,
    ERStrategy,
    EWCStrategy,
    FinetuneStrategy,
    GraspConfig,
    GraspStrategy,
    LinearScheduleConfig,
    Learner,
    LwFStrategy,
    SiestaConfig,
    SiestaStrategy,
    SgmConfig,
    SgmStrategy,
    RichERBuffer,
    accuracy,
    average_accuracy,
    average_last_row,
    average_over_matrix,
    build_cifar10_cil_stream,
    build_cifar100_cil_stream,
    build_resnet18,
    classwise_accuracy_and_confusion,
    classwise_accuracy_over_stream,
    classification_report_from_confusion,
    compute_forgetting,
)


def parse_args():
    p = argparse.ArgumentParser(description="CL experiment with accuracy matrix + per-class/confusion logging")
    # Algorithm selection
    p.add_argument(
        "--algo",
        choices=["er", "finetune", "ewc", "lwf", "agem", "basr", "grasp", "siesta", "sgm", "agm"],
        default="er",
        help="Continual learning algorithm to use",
    )
    p.add_argument("--backend", choices=["clx", "avalanche"], default="clx")
    p.add_argument("--run-name", default=None, help="Optional suffix in result filename.")

    # Dataset configuration
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--n-experiences", type=int, default=None, help="If unset: 5 for CIFAR-10, 10 for CIFAR-100")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="Optional list of seeds; overrides --seed when provided.")
    p.add_argument("--output-dir", default="experiments/results", help="Directory to store JSON logs.")

    # Buffer configuration
    p.add_argument("--buffer-capacity", type=int, default=2000)
    p.add_argument("--replay-ratio", type=float, default=1.0, help="For replay-based methods")

    # Training configuration
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=3)
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


def normalize_algo_name(name: str) -> str:
    """Map aliases to canonical strategy names."""
    aliases = {"agm": "sgm"}
    return aliases.get(name, name)


def get_cifar_transforms(dataset: str):
    del dataset  # same transforms for CIFAR-10/100
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor()])
    return train_transform, test_transform


def build_stream(args, train_transform=None, test_transform=None):
    if args.dataset == "cifar10":
        n_exp = args.n_experiences or 5
        stream = build_cifar10_cil_stream(
            args.data_root, n_experiences=n_exp, seed=args.seed, train_transform=train_transform, test_transform=test_transform
        )
        num_classes = 10
    else:
        n_exp = args.n_experiences or 10
        stream = build_cifar100_cil_stream(
            args.data_root, n_experiences=n_exp, seed=args.seed, train_transform=train_transform, test_transform=test_transform
        )
        num_classes = 100
    return stream, num_classes


def build_strategy(args):
    """Build the CL strategy based on algorithm selection (clx backend)."""
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
    if args.algo == "grasp":
        base = ERStrategy(replay_ratio=args.replay_ratio) if args.replay_ratio > 0 else None
        grasp_cfg = GraspConfig(
            replay_ratio=args.replay_ratio,
            prototype_momentum=0.99,
            difficulty_schedule=LinearScheduleConfig(start=1.0, end=1.0, total_steps=5000),
            min_samples_per_class=2,
            recompute_features_every=200,
        )
        return GraspStrategy(grasp_cfg, base_strategy=base)
    if args.algo == "siesta":
        return SiestaStrategy(SiestaConfig())
    if args.algo == "sgm":
        base = ERStrategy(replay_ratio=args.replay_ratio) if args.replay_ratio > 0 else None
        return SgmStrategy(SgmConfig(), base_strategy=base)
    raise ValueError(f"Unknown algorithm: {args.algo}")


def build_buffer(args):
    """Build the appropriate buffer for the selected algorithm (clx backend)."""
    if args.algo == "finetune":
        return ERBuffer(capacity=100)
    if args.algo == "basr" or args.algo == "grasp":
        return RichERBuffer(capacity=args.buffer_capacity)
    if args.algo == "siesta":
        # Siesta stores latents internally; learner buffer is only used if store_raw_in_er=True.
        return ERBuffer(capacity=args.buffer_capacity)
    return ERBuffer(capacity=args.buffer_capacity)


def class_order_from_stream(experiences: Sequence[Any]) -> List[int]:
    order: List[int] = []
    for exp in experiences:
        for c in getattr(exp, "classes", getattr(exp, "classes_in_this_experience", [])):
            ci = int(c)
            if ci not in order:
                order.append(ci)
    return order


def run_clx_backend(args, train_transform, test_transform) -> Tuple[Dict[str, Any], bool]:
    torch.manual_seed(args.seed)
    fabric = Fabric(accelerator="auto", devices=1, precision=args.precision)
    fabric.launch()

    stream, num_classes = build_stream(args, train_transform, test_transform)
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
    evaluator = ContinualEvaluator(stream, fabric, batch_size=args.batch_size, num_workers=args.num_workers)

    per_exp_acc: List[float] = []
    per_class_acc_by_exp: List[List[Optional[float]]] = []
    confusions: List[List[List[int]]] = []
    prf1_per_exp: List[Dict[str, float]] = []
    stats_log: List[Dict[str, Any]] = []

    fabric.print(f"\n{'='*60}")
    fabric.print(f"Backend: CLX | Algorithm: {args.algo.upper()} | Dataset: {args.dataset.upper()}")
    fabric.print(f"{'='*60}")

    for k, exp in enumerate(stream):
        learner.strategy.before_experience(learner, exp)
        tr_loader, te_loader = learner._make_loaders(exp)
        stats = learner._train_one_experience(tr_loader, exp_id=exp.exp_id)
        learner.strategy.after_experience(learner, exp)

        acc = accuracy(learner.model, te_loader, fabric=fabric)
        per_exp_acc.append(acc)
        acc_row = evaluator.evaluate_after_exp(learner.model, upto_exp=k)
        stats_log.append(stats)

        cls_acc, conf = classwise_accuracy_over_stream(
            learner.model,
            stream,
            k,
            num_classes=num_classes,
            fabric=fabric,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        per_class_acc_by_exp.append(cls_acc)
        confusions.append(conf)
        prf1_per_exp.append(classification_report_from_confusion(conf))

        forgetting_so_far = compute_forgetting(evaluator.acc_matrix)
        cur_forget = sum(forgetting_so_far) / len(forgetting_so_far) if forgetting_so_far else 0.0

        fabric.print(
            f"[Exp {exp.exp_id}] classes={exp.classes} acc={acc:.2f}% "
            f"AA={average_accuracy(per_exp_acc):.2f}% "
            f"updates={stats['num_updates']} time={stats['train_time_sec']:.1f}s "
            f"buffer={stats['buffer_size']}"
        )
        fabric.print(f"  acc_row={['{:.2f}'.format(a) for a in acc_row]} avg_forgetting_so_far={cur_forget:.2f}%")

    forgetting = compute_forgetting(evaluator.acc_matrix)
    avg_forgetting = sum(forgetting) / len(forgetting) if forgetting else 0.0
    avg_acc_final = average_last_row(evaluator.acc_matrix)
    avg_acc_running = average_over_matrix(evaluator.acc_matrix)
    avg_acc_current = average_accuracy(per_exp_acc)
    total_time = sum(s["train_time_sec"] for s in stats_log)
    total_updates = sum(s["num_updates"] for s in stats_log)

    result = {
        "backend": "clx",
        "dataset": args.dataset,
        "algo": args.algo,
        "seed": args.seed,
        "n_experiences": len(stream),
        "num_classes": num_classes,
        "buffer_capacity": args.buffer_capacity,
        "replay_ratio": args.replay_ratio,
        "per_exp_acc": per_exp_acc,
        "acc_matrix": evaluator.acc_matrix,
        "per_class_acc_by_exp": per_class_acc_by_exp,
        "confusions": confusions,
        "prf1_per_exp": prf1_per_exp,
        "class_order": class_order_from_stream(stream),
        "stats_log": stats_log,
        "avg_accuracy": avg_acc_final,
        "avg_accuracy_final": avg_acc_final,
        "avg_accuracy_running": avg_acc_running,
        "avg_accuracy_current_exp": avg_acc_current,
        "avg_forgetting": avg_forgetting,
        "total_time_sec": total_time,
        "total_updates": total_updates,
    }

    return result, fabric.global_rank == 0


def build_avalanche_benchmark(args, train_transform, test_transform):
    try:
        from avalanche.benchmarks.classic import SplitCIFAR10, SplitCIFAR100
    except ImportError as exc:
        raise SystemExit(
            "Avalanche is not installed. Install with `pip install '.[avalanche]'` or add the extra."
        ) from exc

    if args.dataset == "cifar10":
        n_exp = args.n_experiences or 5
        benchmark = SplitCIFAR10(
            n_experiences=n_exp,
            seed=args.seed,
            return_task_id=False,
            dataset_root=args.data_root,
            train_transform=train_transform,
            eval_transform=test_transform,
        )
        num_classes = 10
    else:
        n_exp = args.n_experiences or 10
        benchmark = SplitCIFAR100(
            n_experiences=n_exp,
            seed=args.seed,
            return_task_id=False,
            dataset_root=args.data_root,
            train_transform=train_transform,
            eval_transform=test_transform,
        )
        num_classes = 100
    return benchmark, num_classes


def build_avalanche_strategy(args, model, optimizer, criterion, device):
    try:
        from avalanche.training.supervised import AGEM, EWC, LwF, Naive, Replay
    except ImportError as exc:
        raise SystemExit(
            "Avalanche is not installed. Install with `pip install '.[avalanche]'` or add the extra."
        ) from exc

    supported = {"finetune", "er", "ewc", "lwf", "agem"}
    if args.algo not in supported:
        raise SystemExit(f"Algorithm '{args.algo}' is not supported on the Avalanche backend; use --backend clx.")

    common = dict(train_mb_size=args.batch_size, train_epochs=args.epochs, eval_mb_size=args.batch_size, device=device)

    if args.algo == "finetune":
        return Naive(model, optimizer, criterion, **common)
    if args.algo == "er":
        return Replay(model, optimizer, criterion, mem_size=args.buffer_capacity, **common)
    if args.algo == "ewc":
        return EWC(model, optimizer, criterion, ewc_lambda=args.ewc_lambda, **common)
    if args.algo == "lwf":
        return LwF(model, optimizer, criterion, alpha=args.lwf_alpha, temperature=args.lwf_temperature, **common)
    if args.algo == "agem":
        sample_size = min(args.agem_mem_size, args.batch_size)
        return AGEM(
            model,
            optimizer,
            criterion,
            patterns_per_experience=args.agem_mem_size,
            sample_size=sample_size,
            **common,
        )
    raise ValueError(f"Avalanche backend does not support algo={args.algo}")


def run_avalanche_backend(args, train_transform, test_transform) -> Tuple[Dict[str, Any], bool]:
    torch.manual_seed(args.seed)
    benchmark, num_classes = build_avalanche_benchmark(args, train_transform, test_transform)
    train_stream = list(benchmark.train_stream)
    test_stream = list(benchmark.test_stream)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(num_classes=num_classes, pretrained=False).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    strategy = build_avalanche_strategy(args, model, optimizer, criterion, device)

    per_exp_acc: List[float] = []
    acc_matrix: List[List[float]] = []
    per_class_acc_by_exp: List[List[Optional[float]]] = []
    confusions: List[List[List[int]]] = []
    prf1_per_exp: List[Dict[str, float]] = []
    stats_log: List[Dict[str, Any]] = []

    print(f"\n{'='*60}")
    print(f"Backend: Avalanche | Algorithm: {args.algo.upper()} | Dataset: {args.dataset.upper()}")
    print(f"{'='*60}")

    for k, exp in enumerate(train_stream):
        start_iter = getattr(strategy.clock, "train_iterations", 0)
        t0 = time.perf_counter()
        strategy.train(exp)
        t1 = time.perf_counter()
        end_iter = getattr(strategy.clock, "train_iterations", start_iter)
        num_updates = int(end_iter) - int(start_iter) if isinstance(end_iter, (int, float)) else 0

        # Current experience accuracy
        te_loader = torch.utils.data.DataLoader(
            test_stream[k].dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        acc = accuracy(strategy.model, te_loader, device=device)
        per_exp_acc.append(acc)

        # Accuracy matrix over seen experiences
        row: List[float] = []
        for te in test_stream[: k + 1]:
            loader = torch.utils.data.DataLoader(
                te.dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
            )
            row.append(accuracy(strategy.model, loader, device=device))
        acc_matrix.append(row)

        cls_acc, conf = classwise_accuracy_and_confusion(
            strategy.model,
            [te.dataset for te in test_stream[: k + 1]],
            num_classes=num_classes,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        per_class_acc_by_exp.append(cls_acc)
        confusions.append(conf)
        prf1_per_exp.append(classification_report_from_confusion(conf))

        buffer_size = None
        storage = getattr(strategy, "storage_policy", None)
        if storage is not None and hasattr(storage, "buffer"):
            try:
                buffer_size = len(storage.buffer)
            except Exception:
                buffer_size = getattr(storage, "buffer_size", None)

        stats_log.append(
            {
                "exp_id": int(getattr(exp, "current_experience", k)),
                "num_updates": num_updates,
                "train_time_sec": t1 - t0,
                "buffer_size": buffer_size,
            }
        )

        forgetting_so_far = compute_forgetting(acc_matrix)
        cur_forget = sum(forgetting_so_far) / len(forgetting_so_far) if forgetting_so_far else 0.0

        print(
            f"[Exp {k}] classes={getattr(exp, 'classes_in_this_experience', [])} acc={acc:.2f}% "
            f"AA={average_accuracy(per_exp_acc):.2f}% updates={num_updates} time={t1 - t0:.1f}s "
            f"buffer={buffer_size}"
        )
        print(f"  acc_row={['{:.2f}'.format(a) for a in row]} avg_forgetting_so_far={cur_forget:.2f}%")

    forgetting = compute_forgetting(acc_matrix)
    avg_forgetting = sum(forgetting) / len(forgetting) if forgetting else 0.0
    avg_acc_final = average_last_row(acc_matrix)
    avg_acc_running = average_over_matrix(acc_matrix)
    avg_acc_current = average_accuracy(per_exp_acc)
    total_time = sum(s["train_time_sec"] for s in stats_log)
    total_updates = sum(s["num_updates"] for s in stats_log)

    result = {
        "backend": "avalanche",
        "dataset": args.dataset,
        "algo": args.algo,
        "seed": args.seed,
        "n_experiences": len(train_stream),
        "num_classes": num_classes,
        "buffer_capacity": args.buffer_capacity,
        "replay_ratio": args.replay_ratio,
        "per_exp_acc": per_exp_acc,
        "acc_matrix": acc_matrix,
        "per_class_acc_by_exp": per_class_acc_by_exp,
        "confusions": confusions,
        "prf1_per_exp": prf1_per_exp,
        "class_order": class_order_from_stream(train_stream),
        "stats_log": stats_log,
        "avg_accuracy": avg_acc_final,
        "avg_accuracy_final": avg_acc_final,
        "avg_accuracy_running": avg_acc_running,
        "avg_accuracy_current_exp": avg_acc_current,
        "avg_forgetting": avg_forgetting,
        "total_time_sec": total_time,
        "total_updates": total_updates,
    }

    return result, True


def write_result_json(result: Dict[str, Any], outdir: Path, run_name: Optional[str]) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{run_name}" if run_name else ""
    fname = f"{result['dataset']}_algo-{result['algo']}_backend-{result['backend']}_seed-{result['seed']}{suffix}.json"
    path = outdir / fname
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return path


def main():
    args = parse_args()
    args.algo = normalize_algo_name(args.algo)
    train_transform, test_transform = get_cifar_transforms(args.dataset)

    seeds = args.seeds or [args.seed]
    results = []

    for seed in seeds:
        args.seed = seed
        if args.backend == "clx":
            result, is_primary = run_clx_backend(args, train_transform, test_transform)
        else:
            result, is_primary = run_avalanche_backend(args, train_transform, test_transform)

        out_path = None
        if is_primary:
            out_path = write_result_json(result, Path(args.output_dir), args.run_name)
            print(f"\nSaved results to {out_path}")

        results.append(result)
        print(
            f"[seed {seed}] final-row AA={result['avg_accuracy']:.2f}% "
            f"running AA={result['avg_accuracy_running']:.2f}% "
            f"current-exp AA={result['avg_accuracy_current_exp']:.2f}% "
            f"avg forgetting={result['avg_forgetting']:.2f}% "
            f"total time={result['total_time_sec']:.1f}s backend={result['backend']}"
        )

    if len(results) > 1:
        avg_acc = sum(r["avg_accuracy"] for r in results) / len(results)
        avg_acc_run = sum(r["avg_accuracy_running"] for r in results) / len(results)
        avg_acc_cur = sum(r["avg_accuracy_current_exp"] for r in results) / len(results)
        avg_forgetting = sum(r["avg_forgetting"] for r in results) / len(results)
        print(
            f"\nRan {len(results)} seeds. "
            f"Mean final-row AA={avg_acc:.2f} running AA={avg_acc_run:.2f} "
            f"current-exp AA={avg_acc_cur:.2f} mean avg_forgetting={avg_forgetting:.2f}"
        )


if __name__ == "__main__":
    main()
