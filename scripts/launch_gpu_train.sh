#!/bin/bash
# Generic GPU train pipeline: verify -> smoke -> full SFT -> merge.
# Domain-specific wrappers (e.g. VibeBounty) live in the product repo.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-WeiboAI/VibeThinker-3B}"
DATA="${DATA:-data/train_traces.jsonl}"
ADAPTER_OUT="${ADAPTER_OUT:-adapters/run}"
MERGED_OUT="${MERGED_OUT:-models/merged}"
MIN_TRACES="${MIN_TRACES:-100}"

if ! command -v emberglass-tune >/dev/null 2>&1; then
  echo "[launch] FATAL: install emberglass-tune (pip install -e .)"
  exit 1
fi

N=$(wc -l < "$DATA")
echo "[launch] traces=$N model=$MODEL $(date)"
if [ "$N" -lt "$MIN_TRACES" ]; then
  echo "[launch] FATAL: need >=$MIN_TRACES traces, have $N"
  exit 1
fi

echo "[launch] verify tokenization ..."
emberglass-tune verify --model "$MODEL" --data "$DATA" --min-usable "$MIN_TRACES"

echo "[launch] smoke (8 steps) ..."
emberglass-tune train --model "$MODEL" --data "$DATA" \
  --out adapters/_smoke --limit 128 --max-steps 8 \
  --bs 4 --grad-accum 2 --save-steps 8 --valid-frac 0.05

echo "[launch] full SFT ..."
emberglass-tune train --model "$MODEL" --data "$DATA" \
  --out "$ADAPTER_OUT" --epochs "${EPOCHS:-3}" --bs "${TRAIN_BS:-4}" \
  --grad-accum "${GRAD_ACCUM:-8}" --save-steps "${SAVE_STEPS:-200}"

echo "[launch] merge ..."
emberglass-tune merge --base "$MODEL" --adapter "$ADAPTER_OUT" --out "$MERGED_OUT"

echo "[launch] DONE adapter=$ADAPTER_OUT merged=$MERGED_OUT $(date)"
