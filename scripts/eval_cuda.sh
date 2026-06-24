#!/bin/bash
# CUDA eval smoke: tokenization gate + one greedy generation with base+adapter.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BASE="${BASE:-WeiboAI/VibeThinker-3B}"
ADAPTER="${ADAPTER:-adapters/run}"
DATA="${DATA:-data/train_traces.jsonl}"

if ! command -v emberglass-tune >/dev/null 2>&1; then
  echo "[eval_cuda] FATAL: pip install -e ."
  exit 1
fi

if [ -f "$DATA" ]; then
  emberglass-tune verify --model "$BASE" --data "$DATA" --min-usable 1 --sample 32
fi

python -m emberglass_tune.eval_smoke --base "$BASE" --adapter "$ADAPTER"
echo "[eval_cuda] OK"
