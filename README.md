# Emberglass Tune

**Train LoRA adapters for VibeThinker-3B** (Qwen2.5-class reasoning models) on **MLX** (Apple Silicon) or **CUDA** (Linux GPU). This repo owns the **backward pass** only.

| Repo | Role |
|---|---|
| **[emberglass](https://github.com/maceip/qwen-webgpu-lora)** | Optimized **WebGPU inference** — forward pass, int4, LoRA hot-swap |
| **emberglass-tune** (this) | **Train + eval** — traces, LoRA SFT, merge |
| **[vibebounty](https://github.com/maceip/vibebounty)** | **Product demo** — bug-bounty data, HackerOne UI, serve |

---

## Install

```bash
cd ~/emberglass-tune
uv sync
uv run emberglass-tune --help
```

Optional extras:

```bash
uv sync --extra mlx        # Apple Silicon: mlx_lm.lora / fuse / generate
uv sync --extra anthropic  # trace generation teacher API
```

One-shot without a clone:

```bash
uvx --from git+https://github.com/maceip/emberglass-tune emberglass-tune --help
```

---

## CLI

| Command | Purpose |
|---|---|
| `train` | PEFT LoRA SFT (CUDA / CPU) |
| `merge` | Bake adapter into full base weights |
| `verify` | Tokenization preflight on trace JSONL |
| `gate-traces` | Fail-closed quality gate before training |
| `traces` | Generate teacher `<think>` traces (Anthropic) |
| `eval-smoke` | One-shot generation smoke test |
| `pipeline` | **Unified** verify → smoke → train → merge (hardware presets) |

**Recommended (CUDA):** one command instead of scattered bash scripts:

```bash
uv run emberglass-tune pipeline --preset lambda-gh200 \
  --model ~/models/VibeThinker-3B \
  --data data/sft/train_traces.jsonl \
  --out adapters/run \
  --merged-out ~/models/vibethinker-merged
```

Presets: `lambda-gh200`, `consumer-gpu`, `smoke`. See **[docs/TUNING_2026.md](docs/TUNING_2026.md)** for 2026 research, Lambda optimizations, and alternatives (GRPO, packing, DoRA).

```bash
uv run emberglass-tune train --model WeiboAI/VibeThinker-3B \
  --data traces.jsonl --out adapters/run

uv run emberglass-tune merge --base WeiboAI/VibeThinker-3B \
  --adapter adapters/run --out models/merged
```

---

## Data preparation

Training expects **chat JSONL**: each row has a `messages` array (`system` / `user` / `assistant`). The assistant target for VibeThinker must be a **reasoning trace + JSON verdict**, not bare labels:

```
<think>
{analyst reasoning from report evidence}
</think>
{"disposition":"valid_impactful","severity_estimate":"high",...}
```

### Generic workflow

1. **Build labeled rows** — your domain labels + prompts in chat format (`messages`).
2. **Generate traces** (recommended for reasoning models) — `emberglass-tune traces` (see below).
3. **Gate traces** — `emberglass-tune gate-traces` before spending GPU hours.
4. **Verify tokenization** — `emberglass-tune verify` (catches truncated assistant tails).
5. **Train** — MLX or CUDA path below.
6. **Merge** (optional) — `emberglass-tune merge` for vLLM / simpler serve.

### VibeBounty (bug-bounty triage)

Domain-specific dataset prep lives in **[vibebounty](https://github.com/maceip/vibebounty)**:

| Step | Where | Output |
|---|---|---|
| Label corpus → SFT rows | `vibebounty/data/build_sft.py`, `rebuild_sft_from_jsonl.py` | `data/sft/train.jsonl`, `valid.jsonl`, `test.jsonl` |
| Seed high-value rows for traces | `vibebounty/data/select_trace_seed.py` | `train_trace_seed.jsonl` |
| Teacher traces | `emberglass-tune traces` (this repo) | `train_traces.jsonl` |
| Product prompts / rubric | `vibebounty/prompts/`, `rubric.md` | — |

Train from vibebounty after `uv sync`:

```bash
cd ~/vibebounty
uv run emberglass-tune traces --in data/sft/train.jsonl --out data/sft/train_traces.jsonl
uv run emberglass-tune gate-traces --traces data/sft/train_traces.jsonl --test data/sft/test.jsonl --out ops/trace_gate.json
uv run emberglass-tune train --model WeiboAI/VibeThinker-3B --data data/sft/train_traces.jsonl --out adapters/run
```

---

## Advanced tune: Anthropic teacher + judge

VibeThinker is a **reasoning model**. A cold-start tune on short JSON-only targets collapses — the model needs long, faithful `<think>` traces that reason **forward from report evidence**, not backward from the label.

### Phase 1 — Teacher traces (`traces`)

`emberglass-tune traces` calls the **Anthropic API** (default model: `claude-opus-4-8`) as a **teacher**:

- Input: existing SFT row (system prompt + report + gold JSON verdict).
- Teacher writes private analyst reasoning, ending with `VERDICT: <disposition>`.
- Output assistant target:

  ```
  <think>
  {teacher reasoning}
  </think>
  {gold_json}
  ```

**Quality gates** (fail-closed; retry up to `--retries`, then drop):

| Gate | What it checks |
|---|---|
| **Consistency** | Teacher's `VERDICT:` line matches gold disposition |
| **Leakage** | No "gold label", "I was told", "resolved as", bounty outcome language, etc. |
| **Grounding** | Trace cites this report (≥3 non-generic content words from the submission) |
| **Length** | ≥200 chars of reasoning |

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv sync --extra anthropic

uv run emberglass-tune traces \
  --in data/sft/train.jsonl \
  --out data/sft/train_traces.jsonl \
  --workers 8
```

### Phase 2 — Evidence-only judge (`--verify` / `--drop-unfaithful`)

The same Anthropic client can run a **blind judge**: given only the report (no gold shown), predict the disposition from evidence alone (`PREDICT_SYSTEM`).

| Flag | Behavior |
|---|---|
| `--verify` | Record agreement between blind judge and gold label |
| `--drop-unfaithful` | **Drop** rows where blind judge ≠ gold (labels not supported by report text) |

This removes outcome-derived or weak labels before trace generation/training:

```bash
uv run emberglass-tune traces \
  --in data/sft/train.jsonl \
  --out data/sft/train_traces_highconf.jsonl \
  --verify --drop-unfaithful \
  --workers 8
```

### Phase 3 — Trace gate (`gate-traces`)

Before training, `gate-traces` enforces corpus-level stats (long thinking blocks, per-class counts, leakage scan). Used in production runs with thresholds like `--min-traces 1000`, `--min-think-p50 900`.

```bash
uv run emberglass-tune gate-traces \
  --traces data/sft/train_traces.jsonl \
  --test data/sft/test.jsonl \
  --out ops/trace_gate.json \
  --min-traces 1000
```

### Phase 4 — Sanitize assistant leakage

Post-process JSONL to strip assistant-side leakage patterns (182 → 0 rows in the **highconf-sanitized-20260623** run). Scripting lives in vibebounty ops; output file:

`data/sft/train_traces_highconf_sanitized.jsonl` (474 clean rows).

### What we actually shipped (2026-06-23 run)

| Item | Detail |
|---|---|
| Hardware | NVIDIA GH200 (Lambda), base at `VibeThinker-3B` |
| Trainer | `emberglass-tune train` (PEFT, rank 32/alpha 64 on GPU path) |
| Data | `train_traces_highconf_sanitized.jsonl` after Anthropic traces + faithfulness filter + sanitization |
| Adapter | `highconf-sanitized-20260623` (~228 MB) |
| HF | [macmacmacmac/vibebounty](https://huggingface.co/macmacmacmac/vibebounty) |
| Eval | vibebounty `eval/run_eval.py` + deterministic defense layer (not trained) |

See `vibebounty/ops/highconf_sanitized_20260623_status.md` for held-out metrics.

---

## MLX path (Apple Silicon)

Best for **128 GB** Mac-class machines. Uses `mlx_lm` with `mask_prompt: true` (loss only on assistant turn).

```bash
uv sync --extra mlx

# Point configs/lora_default.yaml at your data dir (train.jsonl + valid.jsonl)
# Or copy vibebounty/configs/bugbounty_lora.yaml
mlx_lm.lora --config configs/lora_default.yaml

# Fuse adapter → standalone weights
mlx_lm.fuse --model WeiboAI/VibeThinker-3B \
  --adapter-path adapters --save-path models/mlx-merged

# Smoke eval
mlx_lm.generate --model models/mlx-merged --prompt "Reply with one word: ok" --max-tokens 32
```

**Default LoRA hyperparams** (`configs/lora_default.yaml`):

- Base: `WeiboAI/VibeThinker-3B`
- Rank **16**, scale **20**, **all 36 layers** (q/k/v/o + MLP)
- `max_seq_length: 4096`, `iters: 2000`, cosine LR with 150-step warmup
- `mask_prompt: true` — critical for long report prompts

From vibebounty: `bash remote/train.sh` (uses `configs/bugbounty_lora.yaml`).

---

## CUDA path (Linux GPU)

Uses HuggingFace **PEFT** via `emberglass-tune train` or the unified **`pipeline`** command.

**Standard entry (GH200 / Lambda):**

```bash
export TOKENIZERS_PARALLELISM=false
uv run emberglass-tune pipeline --preset lambda-gh200 \
  --model ~/models/VibeThinker-3B \
  --data data/sft/train_traces.jsonl \
  --out adapters/run \
  --merged-out ~/models/vibethinker-merged
```

The `lambda-gh200` preset enables **length bucketing** (less padding waste on variable report lengths), **tokenized disk cache** (skip re-tokenizing on restarts), **bs=8 / grad_accum=4**, and less frequent eval steps.

Manual steps (if you need fine control):

```bash
uv sync

# 1. Preflight tokenization
uv run emberglass-tune verify \
  --model ~/models/VibeThinker-3B \
  --data data/sft/train_traces.jsonl \
  --min-usable 500

# 2. Smoke (8 steps)
uv run emberglass-tune train \
  --model ~/models/VibeThinker-3B \
  --data data/sft/train_traces.jsonl \
  --out adapters/_smoke --limit 128 --max-steps 8

# 3. Full SFT
uv run emberglass-tune train \
  --model ~/models/VibeThinker-3B \
  --data data/sft/train_traces.jsonl \
  --out adapters/run \
  --epochs 3 --bs 4 --grad-accum 8 --save-steps 200

# 4. Merge
uv run emberglass-tune merge \
  --base ~/models/VibeThinker-3B \
  --adapter adapters/run \
  --out ~/models/vibethinker-merged

# 5. Smoke generation
uv run emberglass-tune eval-smoke \
  --base ~/models/VibeThinker-3B \
  --adapter adapters/run
```

From vibebounty: `bash scripts/train_gpu_bugbounty.sh`.

**Serve merged or adapter** via vibebounty `remote/serve_vibethinker.py` or vLLM — see vibebounty README.

---

## LoRA output format

Both paths produce **PEFT-compatible** adapters:

- `adapter_config.json`
- `adapter_model.safetensors`

Use in:

- **Emberglass** — runtime hot-swap in browser (`src/lora_gpu.js`)
- **VibeBounty demo** — `serve_vibethinker.py --adapter …`
- **vLLM / MLX server** — merge first or load adapter at serve time

---

## Project layout

```
emberglass_tune/
  train_sft.py      # CUDA/PEFT trainer
  merge_lora.py     # adapter → merged checkpoint
  trace_gen.py      # Anthropic teacher + blind judge
  trace_gate.py     # pre-train corpus gate
  verify_sft_data.py
  eval_smoke.py
configs/
  lora_default.yaml # MLX template
scripts/
  train_mlx.sh      # mlx_lm.lora wrapper
  launch_gpu_train.sh
  eval_mlx.sh
  eval_cuda.sh
```

---

## Related

- **Inference:** [emberglass](https://github.com/maceip/qwen-webgpu-lora) — `npm run build && npm run serve`
- **Demo:** [vibebounty](https://github.com/maceip/vibebounty) — HackerOne UI + `scripts/serve_local.ps1`
