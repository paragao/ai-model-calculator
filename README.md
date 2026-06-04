# AI Model Training Calculator

Analyze GPU memory, batch configuration, training time, and communication overhead for LLM training on AWS hardware.

Supports dense models (Llama 3.1/3.2/3.3, Qwen 2.5/3/3.5) and MoE models (DeepSeek V3/V3.1/V3.2/V4, Qwen3-MoE, Llama 4 Scout/Maverick) on AWS GPU instances (p5/H100, p5en/H200, p6-b200/B200, p6-b300/B300 Ultra, p6e-gb200/GB200 NVL).

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

## Supported Hardware

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

## Repository Structure

```
model_calculations.py           # Main orchestrator (runs all 6 phases)
configuration/
  variants_config.py            # Model architecture definitions
  hardware_config.py            # AWS GPU instance configs
  project_config.py             # Training parameters
  advanced_config.py            # Expert-level settings
utils/
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
- [Amazon Quick Skill](https://github.com/paragao/amazon-quick-skill) — Desktop AI assistant integration
