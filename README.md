# Emberglass Tune

LoRA / SFT **training** for VibeThinker-3B (MLX + CUDA). Inference: [emberglass](https://github.com/maceip/qwen-webgpu-lora). Demo: [vibebounty](https://github.com/maceip/vibebounty).

## Install & run (uv)

From a clone of this repo:

```bash
cd emberglass-tune
uv sync
uv run emberglass-tune --help
uv run emberglass-tune train --model WeiboAI/VibeThinker-3B --data traces.jsonl --out adapters/run
```

One-shot without installing (from GitHub once pushed):

```bash
uvx --from git+https://github.com/maceip/emberglass-tune emberglass-tune --help
```

From a local path:

```bash
uvx --from /path/to/emberglass-tune emberglass-tune --help
```

Optional extras:

```bash
uv sync --extra mlx        # Apple Silicon MLX
uv sync --extra anthropic  # trace_gen teacher API
```

## Commands

| Command | Purpose |
|---|---|
| `train` | PEFT LoRA SFT (CUDA / CPU) |
| `merge` | Bake adapter into base weights |
| `verify` | Tokenization preflight on JSONL |
| `gate-traces` | Trace quality gate |
| `traces` | Teacher reasoning trace generation |
| `eval-smoke` | One-shot generation smoke test |

## VibeBounty

```bash
cd ../vibebounty
uv sync
uv run emberglass-tune train --model WeiboAI/VibeThinker-3B --data data/sft/train_traces.jsonl --out adapters/run
```
