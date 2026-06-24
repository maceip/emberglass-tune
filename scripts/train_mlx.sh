#!/bin/bash
# MLX LoRA fine-tune (Apple Silicon). Copy configs/lora_default.yaml and edit paths first.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p adapters logs
CONFIG="${1:-configs/lora_default.yaml}"
echo "[train_mlx] config=$CONFIG start $(date)"
mlx_lm.lora --config "$CONFIG" 2>&1 | tee logs/train_mlx.log
echo "[train_mlx] end $(date)"
