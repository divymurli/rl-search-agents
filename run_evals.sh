#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
EVAL_SCRIPT="./supervised/eval_infonce_full1k.py"
DEVICE="${DEVICE:-cuda}"

OUT_DIR="${OUT_DIR:-./eval_runs}"
LOG_DIR="${LOG_DIR:-$OUT_DIR/logs}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# ---- Inputs ----
# One for each checkpoint for each experiment: Choosing the maximal performing checkpoint (by MRR) on the 
# HARD subset of queries
CKPTS=(
  "/home/ubuntu/rl-search-agents/ckpts/baseline_infonce/step22000.pt"
  "/home/ubuntu/rl-search-agents/ckpts/margin_a02_m02/step22000.pt"
  "/home/ubuntu/rl-search-agents/ckpts/margin_a04_m02/step30000.pt"
  "/home/ubuntu/rl-search-agents/ckpts/margin_a08_m02/step30000.pt"
  "/home/ubuntu/rl-search-agents/ckpts/grouped_k2_b10/step30000.pt"
  "/home/ubuntu/rl-search-agents/ckpts/grouped_k4_b10/step30000.pt"
  "/home/ubuntu/rl-search-agents/ckpts/grouped_k6_b10/step30000.pt"
)

JSONLS=(
  "/home/ubuntu/rl-search-agents/data/rl-search-datasets/candidates_dev_1k_rm3_backoff.jsonl"
)

# ---- Output files ----
ts="$(date +%Y%m%d_%H%M%S)"
MASTER_JSON="$OUT_DIR/eval_full_candidates_${ts}.json"

# jq is required
command -v jq >/dev/null 2>&1 || {
  echo "jq is required but not installed." >&2
  exit 1
}

# Initialize master JSON as empty array
echo "[]" > "$MASTER_JSON"

slugify() {
  echo "$1" | sed 's/[\/[:space:]]\+/_/g; s/[^A-Za-z0-9._-]/_/g'
}

echo "Writing all results to:"
echo "  $MASTER_JSON"
echo

for ckpt in "${CKPTS[@]}"; do
  ckpt_base="$(basename "$ckpt")"
  ckpt_tag="$(slugify "${ckpt_base%.*}")"

  for jsonl in "${JSONLS[@]}"; do
    jsonl_base="$(basename "$jsonl")"
    jsonl_tag="$(slugify "${jsonl_base%.*}")"

    tmp_json="$(mktemp)"
    log_file="$LOG_DIR/${ckpt_tag}__${jsonl_tag}__${ts}.log"

    echo "==> ckpt:  $ckpt"
    echo "    jsonl: $jsonl"

    set -x
    "$PYTHON_BIN" "$EVAL_SCRIPT" \
      --ckpt "$ckpt" \
      --jsonl "$jsonl" \
      --device "$DEVICE" \
      --out_json "$tmp_json" \
      2>&1 | tee "$log_file"
    set +x

    # Append tmp_json into master JSON array
    jq --argfile rec "$tmp_json" '. += [$rec]' "$MASTER_JSON" > "${MASTER_JSON}.tmp"
    mv "${MASTER_JSON}.tmp" "$MASTER_JSON"

    rm -f "$tmp_json"
    echo
  done
done

echo "Done."
echo "Final results JSON:"
echo "  $MASTER_JSON"
