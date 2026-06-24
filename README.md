# Emberglass Tune

**Train and smoke-eval LoRA adapters for VibeThinker-3B (Qwen2.5-class) on MLX and CUDA.**

This repo owns the **backward pass** only. It does not run the HackerOne demo or browser WebGPU inference.

| Repo | Role |
|---|---|
| **[emberglass](https://github.com/maceip/qwen-webgpu-lora)** | Optimized **WebGPU inference** for VibeThinker-3B |
| **emberglass-tune** (this) | **Tune + eval** scripts (MLX on Mac, CUDA/PEFT on Linux GPU) |
| **[vibebounty](https://github.com/maceip/vibebounty)** | **Product demo**: bug-bounty tune + HackerOne UI + serve |

## Install

```bash
cd ~/emberglass-tune
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .                   # CUDA / PEFT training
pip install -e ".[mlx]"            # optional: Apple Silicon MLX
```

## CLI

```bash
emberglass-tune train        --model WeiboAI/VibeThinker-3B --data traces.jsonl --out adapters/run
emberglass-tune merge        --base WeiboAI/VibeThinker-3B --adapter adapters/run --out models/merged
emberglass-tune verify       --model WeiboAI/VibeThinker-3B --data traces.jsonl --min-usable 100
emberglass-tune gate-traces  --traces traces.jsonl --test test.jsonl --out gate.json
emberglass-tune traces       --in train.jsonl --out train_traces.jsonl --workers 8
emberglass-tune eval-smoke   --base WeiboAI/VibeThinker-3B --adapter adapters/run
```

## Scripts

| Script | Platform | Purpose |
|---|---|---|
| `scripts/train_mlx.sh` | MLX / Mac | `mlx_lm.lora` with `configs/lora_default.yaml` |
| `scripts/launch_gpu_train.sh` | CUDA | verify → smoke train → full SFT → merge |
| `scripts/eval_mlx.sh` | MLX | fuse + `mlx_lm.generate` smoke |
| `scripts/eval_cuda.sh` | CUDA | verify + `eval-smoke` generation |

## VibeBounty usage

[VibeBounty](https://github.com/maceip/vibebounty) keeps domain data, prompts, and the demo app. Install this package from the sibling directory:

```bash
cd ~/vibebounty
pip install -r requirements-train.txt
bash scripts/train_gpu_bugbounty.sh
```

Domain-specific metrics (9-class triage accuracy, adversarial suite) stay in **vibebounty/eval/**.

## Related

- Inference: [emberglass](https://github.com/maceip/qwen-webgpu-lora)
- Demo app: [vibebounty](https://github.com/maceip/vibebounty)
