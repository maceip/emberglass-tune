#!/bin/bash
# MLX eval smoke: fuse adapter (if needed) + mlx_lm.generate one prompt.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BASE="${BASE:-WeiboAI/VibeThinker-3B}"
ADAPTER="${ADAPTER:-adapters}"
FUSED="${FUSED:-models/mlx-merged}"
PROMPT="${PROMPT:-Reply with one word: ok}"

if ! command -v mlx_lm.generate >/dev/null 2>&1; then
  echo "[eval_mlx] FATAL: pip install -e \".[mlx]\""
  exit 1
fi

if [ ! -f "$FUSED/config.json" ]; then
  echo "[eval_mlx] fusing adapter -> $FUSED"
  mlx_lm.fuse --model "$BASE" --adapter-path "$ADAPTER" --save-path "$FUSED"
fi

mlx_lm.generate --model "$FUSED" --prompt "$PROMPT" --max-tokens 32
echo "[eval_mlx] OK"
