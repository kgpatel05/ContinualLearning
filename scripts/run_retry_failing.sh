#!/usr/bin/env bash
# Rerun only the failing/weak algorithms with stronger configs.
# Adjust SEEDS/DATASET/BACKEND as needed before running.
#
# Usage:
#   bash scripts/run_retry_failing.sh

set -euo pipefail

SEEDS=("0" "1")
DATASET="cifar10"
BACKEND="clx"
OUTDIR="experiments/results"

echo "Rerunning selected algos on ${DATASET} | backend=${BACKEND} | seeds=${SEEDS[*]}"

# ER (stronger replay)
python experiments/run_cl_experiment.py \
  --algo er \
  --dataset "${DATASET}" \
  --backend "${BACKEND}" \
  --seeds "${SEEDS[@]}" \
  --buffer-capacity 4000 \
  --replay-ratio 2.0 \
  --output-dir "${OUTDIR}"

# EWC (add replay, soften lambda)
python experiments/run_cl_experiment.py \
  --algo ewc \
  --dataset "${DATASET}" \
  --backend "${BACKEND}" \
  --seeds "${SEEDS[@]}" \
  --buffer-capacity 4000 \
  --replay-ratio 1.0 \
  --ewc-lambda 100 \
  --output-dir "${OUTDIR}"

# AGEM (larger reference/buffer)
python experiments/run_cl_experiment.py \
  --algo agem \
  --dataset "${DATASET}" \
  --backend "${BACKEND}" \
  --seeds "${SEEDS[@]}" \
  --buffer-capacity 4000 \
  --agem-mem-size 512 \
  --output-dir "${OUTDIR}"

echo "Done. Outputs in ${OUTDIR}"
