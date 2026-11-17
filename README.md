# AI Model Training Calculator

Analyze GPU memory, batch configuration, training time, and communication overhead for LLM training on AWS hardware.

Supports dense models (Llama 3.1/3.2/3.3, Qwen 2.5/3/3.5) and MoE models (DeepSeek V3/V3.1/V3.2/V4, Qwen3-MoE, Llama 4 Scout/Maverick) on AWS GPU instances (p5/H100, p5en/H200, p6-b200/B200, p6-b300/B300 Ultra, p6e-gb200/GB200 NVL72). Other platforms can be supplied with `--hardware`.

## Modes of Operation

| Mode | Question | Required Inputs |
|------|----------|----------------|
| **A** | How many instances do I need to finish in time X? | Model, dataset tokens, target time |
| **B** | How long will training take? | Model, dataset tokens, instance count |
| **C** | What's the minimum hardware to fit this model? | Model, instance type |

## What It Calculates

| Phase | Output |
|-------|--------|
| 1. Memory Analysis | Per-GPU memory breakdown with MoE dispatch buffers, VP overhead, NCCL scaling, and fragmentation factor |
| 2. Batch Configuration | Optimal micro batch size and gradient accumulation steps |
| 3. Training Time | Wall-clock estimates with low/high confidence ranges |
| 4. ZeRO Communication | Reduce-scatter (ZeRO-2) and all-gather (ZeRO-1) latency |
| 5. MoE All-to-All | Intra-node communication for expert routing |
| 6. PP SendRecv | Pipeline parallelism latency — NVLink (intra-node) vs EFA (inter-node), bubble % |

### Phase 9: estimated MFU, plan search, days and cost

`--estimate` replaces the fixed `MFU = 0.40` with an estimate from the architecture, the parallel plan
and the hardware, and searches every TP x PP x EP plan that divides the GPU count (memory checked with
that plan). It prints the plan, MFU, tokens/s, days (90% goodput), cost and the parts of the step time.

```bash
python3 ai_model_calculator.py --hf-config config.json --estimate --instance h200 --num-gpus 64 --tokens-b 1000
python3 ai_model_calculator.py --model Llama-3.1-70B --estimate --num-gpus auto --target-days 30 --tokens-b 15000
```

Options: `--instance`, `--num-gpus N|auto`, `--tokens-b`, `--seq-len`, `--precision bf16|fp8`,
`--tp/--pp/--ep` (fix a degree), `--vp`, `--gbs`, `--mfu` (override), `--target-days`, `--format text|json`.
Formulas, constants and sources: [docs/CALCULATOR-MFU.md](docs/CALCULATOR-MFU.md). Accuracy against
29 published runs: `python3 -m utils.validation_runs` (median error 5.6%, max 31.0%). The six-phase
analysis still uses `project_config.MFU`.

## Key Features

- **MoE-aware memory**: Dispatch buffers, NCCL workspace scaling with EP (charged only when world size > 1), measured allocator overhead (1.02×), EP/DP optimizer sharding
- **Virtual Pipeline Parallelism (VP)**: In-flight micro-batch activation overhead modeling
- **Pipeline communication (Phase 6)**: NVLink vs EFA send/recv with bubble percentage and exposed latency
- **Multi-platform comparison**: Sweep node counts across hardware platforms
- **CSV/JSON export**: All phases produce structured exports for analysis

## Getting Started

### Standalone (Python)

```bash
git clone https://github.com/paragao/ai-model-calculator.git
cd ai-model-calculator

# Analyze all built-in variants
python3 ai_model_calculator.py

# Analyze a single built-in variant (name is case-insensitive)
python3 ai_model_calculator.py --model llama-3.1-8b

# Analyze a custom architecture from a JSON file (replaces built-in variants)
python3 ai_model_calculator.py --json my_model.json

# Analyze a model straight from its Hugging Face config.json (parameters, shared
# experts, MLA and linear-attention layers are derived; --params-total-b is the
# exact total from the safetensors index, optional)
python3 ai_model_calculator.py --hf-config config.json --params-total-b 321.32

# Analyze against a custom hardware set from a JSON file (replaces built-in hardware)
python3 ai_model_calculator.py --hardware my_hardware.json

# Combine custom model + custom hardware, single entries selected by name
python3 ai_model_calculator.py --json my_model.json --model my-model \
  --hardware my_hardware.json --hw-name my-cluster

# Concise markdown summary instead of full output
python3 ai_model_calculator.py --model qwen3-8b --report
```

`ai_model_calculator.py` is the entry point: it parses the command-line
arguments and calls `model_calculations.py` to do the work. `model_calculations.py`
can still be run directly (`python3 model_calculations.py`), in which case it
analyzes all built-in variants.

No external dependencies — Python standard library only.

#### CLI options

| Option | Description |
|--------|-------------|
| `--model NAME` | Analyze only the variant named `NAME` (case-insensitive, exact match). Applies to the built-in list, or to the `--json` list when combined. Unknown names print the available variants and exit non-zero. |
| `--json FILE` | Analyze the variant(s) defined in `FILE` **only** (replaces the built-in variants). |
| `--hf-config FILE` | Analyze the model described by a Hugging Face `config.json` **only**. Derives total and active parameters (shared experts included), MLA and hybrid linear attention. Multimodal configs are read from `text_config`. Not combinable with `--json`. |
| `--params-total-b B` | With `--hf-config`: exact total parameters in billions. Wins over the count derived from the shape. |
| `--hardware FILE` | Analyze against the hardware config(s) defined in `FILE` **only** (replaces the built-in hardware list). |
| `--hw-name NAME` | Analyze only the hardware entry named `NAME` (case-insensitive, exact match). Applies to the built-in list, or to the `--hardware` list when combined. Unknown names print the available hardware and exit non-zero. |
| `--report` | Emit a concise markdown-style summary instead of the full per-phase output. |

#### Custom model JSON format

A JSON file may contain a single variant object, a bare list of variants, or a
`{"variants": [ ... ]}` wrapper. Supply the **base** architecture fields; derived
fields (`attn_per_layer`, `expert_each`, `shared_each`, `router_each`,
`moe_layer_params`, `dense_layer_params`) are computed automatically.

Copy [`model_variant_template.json`](model_variant_template.json) as a starting
point — it ships with a field guide plus one dense and one MoE example.

Required base fields: `name`, `layers`, `d`, `dense_layers`, `total_params_B`,
`active_params_B`, `valid_pp`. MoE variants also provide `expert_ffn`,
`dense_ffn`, `q_heads`, `kv_heads`, and may set `num_experts` per variant
(otherwise the project-level `N_EXPERTS` is used). Optionally set `seq_len` per
variant (defaults to the project-level `SEQ_LEN`); it drives memory,
communication, tokens/batch, and the O(seq_len²) attention term in training
time, so a 32K-context variant estimates a longer time than a 4K one at equal
total tokens.

```json
{
  "variants": [
    {
      "name": "MyCustom-MoE-60B-A6B",
      "layers": 48,
      "d": 4096,
      "expert_ffn": 1024,
      "dense_ffn": 0,
      "q_heads": 32,
      "kv_heads": 4,
      "dense_layers": 0,
      "num_experts": 64,
      "total_params_B": 60.0,
      "active_params_B": 6.0,
      "valid_pp": [1, 2, 3, 4, 6, 8, 12, 16, 24, 48]
    }
  ]
}
```

#### Custom hardware JSON format

A JSON file may contain a single hardware object, a bare list, or a
`{"hardware": [ ... ]}` wrapper. Each entry requires: `name`, `gpus`, `nodes`,
`mem_gb`, `inter_node_bw_gb`, `intra_node_bw_gbps`, `peak_tflops_bf16`.
`label` is accepted but unused; `gpus_per_node` is assumed to be 8. Note that
`gpus` must be divisible by `TP*PP*CP` (from `project_config.py`) or validation
fails.

Copy [`hardware_variant_template.json`](hardware_variant_template.json) as a
starting point — it ships with a field guide plus two example clusters.

```json
{
  "hardware": [
    {
      "name": "MyCluster-Large",
      "gpus": 512,
      "nodes": 64,
      "mem_gb": 141,
      "inter_node_bw_gb": 400,
      "intra_node_bw_gbps": 900,
      "peak_tflops_bf16": 989
    }
  ]
}
```

### Docker

```bash
cd container
docker compose up
```

See `container/README.md` for details.

### As a Kiro Power

Install via Kiro command palette → "Kiro: Add Power":

```
https://github.com/paragao/ai-model-calc-power.git
```

### As an Amazon Quick Skill

```bash
git clone https://github.com/paragao/ai-model-calc-skill.git ~/.quickwork/profiles/federate-prod/skills/ai-model-calc
```

### As a Claude Code Skill

Clone into your personal skills directory (or `.claude/skills/` inside a project to scope it to that repo):

```bash
git clone https://github.com/paragao/ai-model-calc-cc-skill.git ~/.claude/skills/ai-model-calc
```

The skill shells out to this calculator, so clone the tool itself as well and tell the skill where it lives.

## Configuration

Edit `configuration/project_config.py` before running:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VOCAB` | 151,936 | Vocabulary size. **Fallback** -- a variant's own `vocab` wins |
| `SEQ_LEN` | 4,096 | Sequence length |
| `N_EXPERTS` | 128 | Total experts (0 for dense). **Fallback** -- a variant's own `num_experts` wins |
| `TOPK` | 8 | Experts per token. **Fallback** -- a variant's own `topk` wins (Llama-4 Scout/Maverick route to 1, Qwen3.5-397B-A17B to 10) |
| `PP` | 8 | Pipeline parallelism |
| `VP` | 4 | Virtual pipeline parallelism (1 = disabled) |
| `TP` | 1 | Tensor parallelism |
| `EP` | 8 | Expert parallelism (1 for dense) |
| `TOTAL_TOKENS` | 15e12 | Dataset size in tokens |
| `PRECISION` | BF16 | Training precision (BF16 or FP8) |
| `MFU` | 0.40 | Model FLOPs Utilization (0.35–0.55 typical) |

## Supported Hardware

Built-in cluster presets (28 in total), selectable with `--hw-name`. Preset names are `"<nodes> <platform>"` — for example `--hw-name "128 p5en"`. Node counts are 64, 128, 256 and 512 for p5/p5en/p6-b200/p6-b300, and 1, 4, 8 and 16 for g5/g6/g6e:

| Platform | GPU | Memory | Peak BF16 TFLOPS |
|----------|-----|--------|-----------------|
| p5.48xlarge | H100 SXM | 80 GB | 989 |
| p5en.48xlarge | H200 SXM | 141 GB | 989 |
| p6-b200.48xlarge | B200 | 180 GB | 2,250 |
| p6-b300.48xlarge | B300 Ultra | 268 GB | 2,250 |
| g5.48xlarge | A10G | 24 GB | 35 |
| g6.48xlarge | L4 | 24 GB | 121 |
| g6e.48xlarge | L40S | 48 GB | 366 |
| p6e-gb200 (`u-p6e-gb200x72` UltraServer: 72 GPUs, one NVLink domain) | GB200 | 185 GB | 2,500 |

B300 BF16 is 2,250, the same as B200: Blackwell Ultra's 1.5x uplift is FP4 only (see the note in `configuration/hardware_config.py`). Every figure is dense (no sparsity).

GB200 is a built-in preset. The calculator sizes the UltraServer (72 GPUs, or `u-p6e-gb200x36` with 36), the unit EC2 Capacity Blocks sell; a `p6e-gb200.36xlarge` instance is 4 GPUs of it. Run it with `--hw-name "1 p6e-gb200"` (phases 1-8) or `--estimate --instance gb200` (phase 9). AWS publishes no on-demand price for it, so the estimate shows no cost. Sources for every GB200 and GB300 number: `docs/CALCULATOR-HARDWARE.md`.

## Repository Structure

```
ai_model_calculator.py          # CLI entry point (parses args, selects variants)
model_calculations.py           # Analysis engine — run(variants) over all 6 phases
configuration/
  variants_config.py            # Model architecture definitions
  hardware_config.py            # AWS GPU instance configs
  project_config.py             # Training parameters
  advanced_config.py            # Expert-level settings
utils/
  variant_loader.py             # JSON variant loading + name selection
  hardware_loader.py            # JSON hardware loading + name selection
  core_calculations.py          # Shared calculation helpers
  phase1_memory.py              # Memory analysis
  phase2_batch.py               # Batch configuration
  phase3_training.py            # Training time
  phase4_zero2_comm.py          # ZeRO-2 reduce-scatter
  phase4_1_zero1_comm.py        # ZeRO-1 all-gather
  phase5_alltoall_comm.py       # MoE all-to-all
  phase6_pp_comm.py             # PP send/recv communication
  validate_moe_memory.py        # MoE memory validation
  validation.py                 # Input validation
container/                      # Docker distribution
tests/                          # Edge case tests
```

## Related

- [Kiro Power](https://github.com/paragao/ai-model-calc-power) — AI IDE integration
- [Amazon Quick Skill](https://github.com/paragao/ai-model-calc-skill) — Desktop AI assistant integration
- [Claude Code Skill](https://github.com/paragao/ai-model-calc-cc-skill) — Claude Code / terminal agent integration
