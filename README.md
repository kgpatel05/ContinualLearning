# clx_mvp

Small research codebase for class-incremental learning on CIFAR-style benchmarks. Training runs through [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) so device placement and mixed precision stay in one place instead of being hand-rolled in every script.

The core loop lives in `Learner`: you pass an ordered list of `Experience` objects (train/test splits per phase), an `ERBuffer` for replay, and optionally a `CLStrategy` hook that runs before/after each experience. Evaluation in the default path is the usual “test on the current experience after training it”; the experiment runner can log fuller matrices and confusion data if you need paper-style tables.

## Setup

Python 3.10+.

With [uv](https://github.com/astral-sh/uv) (recommended):

```bash
cd ContinualLearning
uv sync
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Editable install with pip works too:

```bash
pip install -e .
```

Optional groups (see `pyproject.toml`):

- `bench` — Weights & Biases
- `avalanche` — run the same experiment driver against Avalanche as a backend
- `dev` — pytest, coverage, ruff, mypy

## Minimal example

```python
from lightning.fabric import Fabric
from clx_mvp import (
    build_cifar10_cil_stream,
    build_resnet18,
    ERBuffer,
    Learner,
)

fabric = Fabric(accelerator="auto", devices=1, precision="bf16-mixed")
fabric.launch()

stream = build_cifar10_cil_stream(data_root="./data", n_experiences=5, seed=0)
model = build_resnet18(num_classes=10, pretrained=False)
buffer = ERBuffer(capacity=2000)

learner = Learner(model=model, fabric=fabric, buffer=buffer, replay_ratio=0.5)
report = learner.fit(stream)

fabric.print(f"Per-experience acc: {report.per_exp_acc}")
fabric.print(f"Average acc: {report.avg_acc:.2f}%")
```

If `bf16-mixed` fails on your GPU, try `16-mixed` or drop the precision argument.

## What’s in the box

**Streams.** `build_cifar10_cil_stream` and `build_cifar100_cil_stream` produce class-IL splits. There are also helpers for generic class-IL, domain-IL, and custom streams in `clx_mvp/streams.py` if you’re prototyping beyond the default benchmarks.

**Strategies.** The learner accepts any `CLStrategy` implementation. Included: experience replay (`ERStrategy`), finetuning-only, EWC, LwF, A-GEM, BASR, GRASP, SIESTA, and SGM—wired so you can swap algorithms without rewriting the training loop.

**Buffers.** Beyond plain reservoir replay (`ERBuffer`), there are richer buffer variants and latent replay pieces (`RichERBuffer`, `LatentReplayBuffer`) for experiments that compress or store representations instead of raw images.

**Metrics.** Accuracy, average accuracy, forgetting, classwise confusion, and a few efficiency helpers (parameter counts, rough FLOPs/memory estimates) live under `clx_mvp/metrics.py`.

**Extras.** LoRA config, feature extractors, and latent compression utilities are there for runs that touch efficient adaptation rather than full fine-tuning only.

## Scripts

| Path | Purpose |
|------|---------|
| `scripts/train_cifar10.py` | Short end-to-end CIFAR-10 run with default hyperparameters. |
| `scripts/demo.py` | Walks through stream builders and optional plotting of per-experience class counts. |

## Experiments

`experiments/run_cl_experiment.py` is the heavier entrypoint: pick `--algo`, `--dataset` (cifar10 / cifar100), buffer size, seeds, and `--backend clx` or `avalanche`. It writes JSON under `experiments/results/` by default. Pair it with `experiments/plot_results.py` to turn those logs into figures.

Example:

```bash
python experiments/run_cl_experiment.py --algo er --dataset cifar10 --n-experiences 5 --backend clx
```

`experiments/run_er_baseline.py` and `experiments/demo_quickstart.py` are additional starting points if you’re comparing baselines or sanity-checking the install.

## Layout

```
ContinualLearning/
├── clx_mvp/
│   ├── learner.py          # Fabric-backed training loop + FitReport
│   ├── streams.py          # CIFAR and generic stream builders
│   ├── replay.py           # Replay buffers
│   ├── models.py           # e.g. ResNet-18 builder
│   ├── metrics.py
│   ├── configs.py
│   ├── compression.py / features.py / lora.py
│   └── strategies/       # CL algorithm hooks
├── experiments/
├── scripts/
└── pyproject.toml
```

## Dependencies

Core: PyTorch, torchvision, Lightning, NumPy, Matplotlib. Exact lower bounds are pinned in `pyproject.toml`.

---

CSC 277 (University of Rochester) — continual learning course project. Code structure may shift as experiments land; treat tagged commits as the stable reference if you’re reproducing numbers.
