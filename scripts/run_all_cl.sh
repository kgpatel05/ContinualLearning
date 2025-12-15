#!/usr/bin/env bash
# Run all CLX algorithms across seeds for CIFAR-10.
# Usage: bash scripts/run_all_cl.sh

set -euo pipefail

ALGOS=("er" "ewc" "lwf" "agem" "basr" "grasp" "siesta" "sgm")
SEEDS=("0" "1" "2")
DATASET="cifar10"
BACKEND="clx"
OUTDIR="experiments/results"

echo "Running CL experiments: backend=${BACKEND}, dataset=${DATASET}"
for algo in "${ALGOS[@]}"; do
  echo "----------------------------------------"
  echo "Algo: ${algo}"
  python experiments/run_cl_experiment.py \
    --algo "${algo}" \
    --dataset "${DATASET}" \
    --backend "${BACKEND}" \
    --seeds "${SEEDS[@]}" \
    --output-dir "${OUTDIR}"
done

echo "Done. Results in ${OUTDIR}"
