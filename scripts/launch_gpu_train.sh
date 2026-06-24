#!/bin/bash
# Generic GPU train pipeline — delegates to emberglass-tune pipeline presets.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL="${MODEL:-WeiboAI/VibeThinker-3B}"
DATA="${DATA:-data/train_traces.jsonl}"
ADAPTER_OUT="${ADAPTER_OUT:-adapters/run}"
MERGED_OUT="${MERGED_OUT:-models/merged}"
PRESET="${PRESET:-lambda-gh200}"

if ! command -v emberglass-tune >/dev/null 2>&1; then
  echo "[launch] FATAL: uv sync && uv run emberglass-tune --help"
  exit 1
fi

N=$(wc -l < "$DATA")
echo "[launch] preset=$PRESET traces=$N model=$MODEL $(date)"

emberglass-tune pipeline --preset "$PRESET" \
  --model "$MODEL" \
  --data "$DATA" \
  --out "$ADAPTER_OUT" \
  --merged-out "$MERGED_OUT"

echo "[launch] DONE adapter=$ADAPTER_OUT merged=$MERGED_OUT $(date)"
