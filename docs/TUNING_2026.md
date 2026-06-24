# Tuning in 2026 — research, alternatives, and Emberglass defaults

This doc summarizes what the 2025–2026 literature recommends for reasoning-model SFT, what we implement today in **emberglass-tune**, and what is on the roadmap.

## What the literature says (2025–2026)

| Technique | Evidence | Relevance to VibeThinker triage |
|---|---|---|
| **LoRA on all linear layers** (attn + MLP) | Consistently beats attention-only LoRA on domain tasks | Already default in CUDA + MLX configs |
| **Rank 16–64 for reasoning** | Raschka / Open-RS GRPO work: r=16 sweet spot; harder domains need 32–64 | We use r=32 CUDA, r=16 MLX (Mac headroom) |
| **α ≈ r or 2r** | Fixed small α with large r under-utilizes capacity | CUDA: α=64 with r=32; MLX scale=20 with r=16 |
| **LR ~2e-4 for LoRA** (10× FFT LR) | Standard PEFT guidance | Default 1e-4 with cosine + warmup |
| **Sequence packing + FlashAttention** | 2×–6× throughput on variable-length SFT (IBM, NeMo, ACL 2025 packing analysis) | **Partially adopted** — length bucketing now; full packing planned |
| **GRPO / RL on LoRA** | Strong on math reasoning transfer (DeepSeek-R1 line) | **Alternative path** — not in repo yet; see below |
| **DoRA / rsLoRA** | Better stability at high rank | Optional experiment (`use_rslora` in PEFT) |
| **Teacher traces before SFT** | Required for reasoning models that emit `<think>` | Anthropic teacher + blind judge in `traces` |
| **Quality gates > more data** | 7k high-quality > 100k noisy (multiple 2025 reasoning papers) | `gate-traces`, `--drop-unfaithful`, sanitization |

References worth reading:

- ACL 2025 Findings — *Packing Analysis* (when packing helps vs hurts)
- IBM Research — TRL `padding_free` + FlashAttention 2 collator
- NeMo 25.04 — cu_seqlens packing for SFT/PEFT
- Open-RS / GRPO + LoRA reasoning transfer (2025)

## What was slow on Lambda (and fixes)

| Symptom | Root cause | Fix in emberglass-tune |
|---|---|---|
| GPU under-utilized, long steps | Variable-length reports → heavy **padding** in batches | **`--bucket-batch`** (length-grouped batches) |
| Long startup before step 1 | Re-tokenizing 10k+ rows every run | **`--cache-dir .emberglass-cache`** |
| Eval every N steps stalls training | Eval on 8k context is expensive | **`--eval-steps`** (2× save interval on GH200 preset) |
| Micro-batch too small for GH200 | Conservative bs=4, accum=8 | **`lambda-gh200` preset**: bs=8, accum=4 |
| Attention not using FA kernels | Hard-coded sdpa only | **`--attn auto`** (flash_attn2 when installed, else sdpa) |
| Scattered bash entrypoints | Many copies of verify/smoke/train | **`emberglass-tune pipeline --preset …`** |

## Standard way to tune (one command)

### CUDA — Lambda GH200

```bash
cd ~/bbverifier   # or vibebounty
export TOKENIZERS_PARALLELISM=false

uv run emberglass-tune pipeline --preset lambda-gh200 \
  --model ~/models/VibeThinker-3B \
  --data data/sft/train_traces_highconf_sanitized.jsonl \
  --out adapters/run \
  --merged-out ~/models/vibethinker-merged
```

Dry-run (print steps only):

```bash
uv run emberglass-tune pipeline --preset lambda-gh200 \
  --model ~/models/VibeThinker-3B \
  --data data/sft/train_traces.jsonl \
  --out adapters/run --dry-run
```

### CUDA — consumer GPU (24 GB)

```bash
uv run emberglass-tune pipeline --preset consumer-gpu \
  --model WeiboAI/VibeThinker-3B \
  --data data/sft/train_traces.jsonl \
  --out adapters/run
```

### MLX — Apple Silicon

MLX stays a single config file (no PEFT Trainer):

```bash
uv sync --extra mlx
mlx_lm.lora --config configs/lora_default.yaml
mlx_lm.fuse --model WeiboAI/VibeThinker-3B \
  --adapter-path adapters --save-path models/mlx-merged
```

Or from vibebounty: `bash remote/train.sh`.

### Presets

| Preset | Hardware | bs × accum | max_len | Notes |
|---|---|---|---|---|
| `lambda-gh200` | GH200 / large CUDA | 8 × 4 | 8192 | bucket + cache + less frequent eval |
| `consumer-gpu` | 24 GB | 2 × 16 | 4096 | safe defaults |
| `smoke` | any | 2 × 2 | 8192 | verify + 8 steps only |

Edit presets in `configs/presets/*.yaml` — no bash duplication.

## Advanced trace pipeline (unchanged, still recommended)

1. **`traces`** — Anthropic teacher → `<think>` + gold JSON  
2. **`--verify --drop-unfaithful`** — blind judge drops unfaithful labels  
3. **`gate-traces`** — corpus-level fail-closed gate  
4. **Sanitize** — strip assistant leakage (vibebounty ops)  
5. **`pipeline --preset lambda-gh200`** — train  

Pilot traces with a cheaper model:

```bash
uv run emberglass-tune traces --in data/sft/train.jsonl \
  --out data/sft/train_traces_pilot.jsonl \
  --model claude-sonnet-4-20250514 --sample-per-class 3 --workers 4
```

## Alternatives to consider (not implemented)

| Alternative | When | Trade-off |
|---|---|---|
| **TRL SFTTrainer + `padding_free=True`** | Next throughput push | Needs TRL dep; best packing story with FA2 |
| **GRPO / Dr. GRPO on LoRA** | Behavior transfer beyond SFT | Needs reward model / verifiable tasks; heavier infra |
| **DoRA** | High-rank runs without rsLoRA | Slightly more VRAM |
| **Unsloth** | Consumer GPU one-liner | Less control over our trace-masking logic |
| **QLoRA 4-bit base** | VRAM-bound 7B+ | Not needed for 3B on GH200 |
| **Distillation from Claude** | Skip trace gen API cost | Loses faithfulness gates we rely on |

Recommended order if we invest further:

1. TRL padding-free packing (biggest remaining CUDA win)  
2. Optional GRPO stage on accept/reject + severity calibration  
3. DoRA/rsLoRA ablation at r=64 for rare classes  

## Environment hygiene (Lambda lessons)

From `vibebounty/ops/software_catalog_for_tune_2026-06-23.md`:

- **Separate venvs** for train (`vt-train`) vs vLLM eval — do not let vLLM pip upgrades break torch.  
- **Smoke before full run** — pipeline always runs 8-step smoke first.  
- **Token cache** — second run on same data should skip tokenization.  
- **Do not mix** training and vLLM repair in one environment mid-run.

## Measuring improvement

Log these before/after preset changes:

```bash
# During train — watch HF logs for samples/sec and loss
# After train
uv run emberglass-tune eval-smoke --base ~/models/VibeThinker-3B --adapter adapters/run
```

On vibebounty held-out:

```bash
python eval/run_eval.py --report eval/report.json --workers 8
```

Compare **tokens/sec**, **wall time per epoch**, and **held-out macro-F1** — not just loss.
