# AI Model Training Calculator

Analyze GPU memory, batch configuration, training time, and communication overhead for LLM training on AWS hardware.

Supports dense models (Llama 3.1/3.2/3.3, Qwen 2.5/3/3.5) and MoE models (DeepSeek V3/V3.1/V3.2/V4, Qwen3-MoE, Llama 4 Scout/Maverick) on AWS GPU instances (p5/H100, p5en/H200, p6-b200/B200, p6-b300/B300 Ultra, p6e-gb200/GB200 NVL).

Now includes an **LLM Inference Calculator** for analyzing serving workload performance (TTFT, decode throughput, ITL, cost efficiency) across AWS GPU instances.

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

## Key Features

- **MoE-aware memory**: Dispatch buffers, NCCL workspace scaling with EP, fragmentation factor (1.24×), EP/DP optimizer sharding
- **Virtual Pipeline Parallelism (VP)**: In-flight micro-batch activation overhead modeling
- **Pipeline communication (Phase 6)**: NVLink vs EFA send/recv with bubble percentage and exposed latency
- **Multi-platform comparison**: Sweep node counts across hardware platforms
- **CSV/JSON export**: All phases produce structured exports for analysis

## Getting Started

### Standalone (Python)

```bash
git clone https://github.com/paragao/ai-model-calculator.git
cd ai-model-calculator
python3 model_calculations.py
```

No external dependencies — Python standard library only.

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
git clone https://github.com/paragao/amazon-quick-skill.git ~/.quickwork/profiles/federate-prod/skills/ai-model-calc
```

## Configuration

Edit `configuration/project_config.py` before running:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VOCAB` | 151,936 | Vocabulary size |
| `SEQ_LEN` | 4,096 | Sequence length |
| `N_EXPERTS` | 128 | Total experts (0 for dense) |
| `TOPK` | 8 | Experts per token |
| `PP` | 8 | Pipeline parallelism |
| `VP` | 4 | Virtual pipeline parallelism (1 = disabled) |
| `TP` | 1 | Tensor parallelism |
| `EP` | 8 | Expert parallelism (1 for dense) |
| `TOTAL_TOKENS` | 15e12 | Dataset size in tokens |
| `PRECISION` | BF16 | Training precision (BF16 or FP8) |
| `MFU` | 0.40 | Model FLOPs Utilization (0.35–0.55 typical) |

## Inference Calculator

Analyze LLM serving performance: KV cache sizing, prefill (TTFT), decode throughput/ITL, TP/EP communication overhead, scheduler batching, and cost efficiency.

### Quick Start

```bash
# Full interactive analysis
python3 inference_calculations.py --variant "Gemma-4-31B-IT" --hw "1 g7e-12xl" --engine vllm

# With custom TP degree
python3 inference_calculations.py --variant "Gemma-4-31B-IT" --hw "1 g6e-48xl" --tp 8

# Markdown report
python3 inference_calculations.py --variant "Gemma-4-31B-IT" --hw "1 g7e-12xl" --report

# All engines comparison
python3 inference_calculations.py --engine all
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--variant NAME` | Filter to a single model variant |
| `--hw NAME` | Filter to a single hardware config |
| `--engine vllm\|sglang\|trtllm\|all` | Override inference engine |
| `--tp N` | Override tensor parallelism degree |
| `--ep N` | Override expert parallelism degree |
| `--gpu-util 0.0-1.0` | Override GPU memory utilization fraction |
| `--report` | Output concise markdown summary |

### Inference Phases

| Phase | Module | Output |
|-------|--------|--------|
| 7. KV Cache & Memory | `phase7_kv_cache.py` | Model memory, KV per request, max concurrent requests |
| 8. Prefill (TTFT) | `phase8_prefill.py` | Time to first token, compute vs memory bound |
| 9. Decode (ITL) | `phase9_decode.py` | Throughput (tok/s), inter-token latency, batched scaling |
| 10. TP Communication | `phase10_tp_comm.py` | All-reduce overhead, NVLink vs PCIe efficiency |
| 11. EP Communication | `phase11_ep_comm.py` | MoE dispatch/combine overhead (MoE models only) |
| 12. Scheduler | `phase12_scheduler.py` | Queue delay, saturation point, effective batch sizing |
| 13. Cost Efficiency | `phase13_cost.py` | $/M tokens, tokens/dollar, optimal operating point |

### Inference Configuration

Edit `configuration/inference_config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INPUT_SEQ_LENS` | [14000, 33000, 49000] | ISL values to sweep |
| `OUTPUT_SEQ_LEN` | 1000 | Output tokens per request |
| `CONCURRENCY_LEVELS` | [1, 8, 32, 64, 128] | Concurrent request levels |
| `ENGINE` | vllm | Inference engine |
| `QUANTIZATION` | fp8 | Weight quantization |
| `KV_CACHE_DTYPE` | fp8 | KV cache precision |
| `GPU_MEMORY_UTILIZATION` | 0.90 | GPU memory fraction for model + KV |

### Validation

Validated against real benchmarks (Gemma-4-31B-IT FP8):

| Config | Metric | Accuracy |
|--------|--------|----------|
| g7e TP=1 | Decode tok/s (conc=1) | 87-100% |
| g7e TP=1 | TTFT | 75-113% |
| g6e TP=8 | TTFT | 100-102% |
| g6e TP=8 | Decode tok/s | Known overestimate (PCIe TP=8 sync stalls not fully modeled) |

Best accuracy on single-GPU or NVLink-connected configurations. PCIe TP>4 decode estimates are directional only.

---

## Supported Hardware

### Training

| Platform | GPU | Memory | Peak BF16 TFLOPS |
|----------|-----|--------|-----------------|
| p5.48xlarge | H100 SXM | 80 GB | 989 |
| p5en.48xlarge | H200 SXM | 141 GB | 989 |
| p6-b200.48xlarge | B200 | 180 GB | 2,250 |
| p6-b300.48xlarge | B300 Ultra | 268 GB | 3,375 |
| p6e-gb200.36xlarge | GB200 NVL | 185 GB | 2,500 |
| g5.48xlarge | A10G | 24 GB | 35 |
| g6.48xlarge | L4 | 24 GB | 121 |
| g6e.48xlarge | L40S | 48 GB | 366 |

### Inference

| Platform | GPU | Memory | HBM BW (GB/s) | FP8 TFLOPS | Interconnect |
|----------|-----|--------|---------------|------------|--------------|
| p5en.48xlarge | H200 SXM | 141 GB | 4,800 | 1,979 | NVLink 900 GB/s |
| p6-b200.48xlarge | B200 | 180 GB | 8,000 | 4,500 | NVLink5 1,800 GB/s |
| p6-b300.48xlarge | B300 Ultra | 268 GB | 12,000 | 6,750 | NVLink5 1,800 GB/s |
| g6e.48xlarge | L40S | 48 GB | 864 | 733 | PCIe Gen4 64 GB/s |
| g7e.48xlarge | L40S v2 | 48 GB | 1,500 | 733 | PCIe Gen5 128 GB/s |
| g7e.12xlarge | RTX PRO 6000 | 96 GB | 1,750 | 2,088 | PCIe Gen5 128 GB/s |

## Repository Structure

```
model_calculations.py           # Training orchestrator (phases 1-6)
inference_calculations.py       # Inference orchestrator (phases 7-13)
configuration/
  variants_config.py            # Model architecture definitions (training + inference)
  hardware_config.py            # AWS GPU instance configs (HARDWARE + INFERENCE_HARDWARE)
  project_config.py             # Training parameters
  inference_config.py           # Inference parameters (ISL, OSL, engine, quantization)
  advanced_config.py            # Expert-level settings
utils/
  core_calculations.py          # Shared calculation helpers
  phase1_memory.py              # Memory analysis (training)
  phase2_batch.py               # Batch configuration (training)
  phase3_training.py            # Training time (training)
  phase4_zero2_comm.py          # ZeRO-2 reduce-scatter (training)
  phase4_1_zero1_comm.py        # ZeRO-1 all-gather (training)
  phase5_alltoall_comm.py       # MoE all-to-all (training)
  phase6_pp_comm.py             # PP send/recv communication (training)
  phase7_kv_cache.py            # KV cache & model memory (inference)
  phase8_prefill.py             # Prefill / TTFT analysis (inference)
  phase9_decode.py              # Decode throughput / ITL (inference)
  phase10_tp_comm.py            # TP communication overhead (inference)
  phase11_ep_comm.py            # EP communication / MoE (inference)
  phase12_scheduler.py          # Scheduler & batching (inference)
  phase13_cost.py               # Cost efficiency (inference)
  formatting_utils.py           # Terminal color formatting
  validate_moe_memory.py        # MoE memory validation
  validation.py                 # Input validation
container/                      # Docker distribution
tests/                          # Edge case tests
```

## Related

- [Kiro Power](https://github.com/paragao/ai-model-calc-power) — AI IDE integration
- [Amazon Quick Skill](https://github.com/paragao/amazon-quick-skill) — Desktop AI assistant integration
