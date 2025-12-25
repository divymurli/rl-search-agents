#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
TRAIN_JSONL="data/rl-search-datasets/msmarco-supervised/training_data/train.jsonl"
DEV_JSONL="data/rl-search-datasets/msmarco-supervised/training_data/dev.jsonl"
SCRIPT="supervised/train_msmarco_supervised_infonce_updated.py"

# ------------------------------------------------------------------
# Common training args (LR + everything else fixed)
# ------------------------------------------------------------------
COMMON_ARGS=(
  --train_jsonl "$TRAIN_JSONL"
  --dev_jsonl   "$DEV_JSONL"
  --batch_size  16
  --accum_steps 4
)

run_exp () {
  local outdir="$1"; shift
  echo "============================================================"
  echo "Running: $outdir"
  echo "python $SCRIPT ${COMMON_ARGS[*]} --outdir $outdir $*"
  echo "============================================================"
  python "$SCRIPT" "${COMMON_ARGS[@]}" --outdir "$outdir" "$@"
}

# ==================================================================
# 0) Baseline: in-batch InfoNCE
# ==================================================================
run_exp "ckpts/baseline_infonce" \
  --loss_mode infonce

# ==================================================================
# 1–3) Margin InfoNCE (fixed m=0.2, sweep alpha)
# ==================================================================
run_exp "ckpts/margin_a02_m02" \
  --loss_mode infonce_margin --margin_alpha 0.2 --margin_m 0.2

run_exp "ckpts/margin_a04_m02" \
  --loss_mode infonce_margin --margin_alpha 0.4 --margin_m 0.2

run_exp "ckpts/margin_a08_m02" \
  --loss_mode infonce_margin --margin_alpha 0.8 --margin_m 0.2

# ==================================================================
# 4–6) Grouped InfoNCE (fixed beta=1.0, sweep k)
# ==================================================================
# run_exp "ckpts/grouped_k2_b10" \
#   --loss_mode infonce_grouped --group_k 2 --group_beta 1.0 --group_temperature 0.05

# run_exp "ckpts/grouped_k4_b10" \
#   --loss_mode infonce_grouped --group_k 4 --group_beta 1.0 --group_temperature 0.05

# run_exp "ckpts/grouped_k6_b10" \
#   --loss_mode infonce_grouped --group_k 6 --group_beta 1.0 --group_temperature 0.05

echo "All experiments completed successfully."
