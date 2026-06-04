---
name: "ai-model-calc"
displayName: "AI Model Training Calculator"
description: "Analyze GPU memory, batch configuration, training time, and communication overhead for LLM training across AWS hardware platforms"
keywords: ["llm training", "gpu memory", "batch size", "training time", "zero communication", "moe", "mixture of experts", "parallelism", "h100", "h200", "p5", "p5en", "model calculator", "infrastructure planning", "capacity planning"]
author: "paragao"
---

# AI Model Training Calculator

## What this power does

This power analyzes training infrastructure requirements for large language models. It runs a 6-phase calculator that computes:

1. **Memory Analysis** -- Per-GPU memory breakdown (model, gradient, optimizer, activation, buffers) across ZeRO stages. Includes MoE-aware estimation with dispatch buffers, VP activation overhead, NCCL EP scaling, and fragmentation factor (1.24×).
2. **Batch Configuration** -- Optimal micro batch size and gradient accumulation for target throughput
3. **Training Time** -- Wall-clock time estimates with low/high confidence ranges
4. **ZeRO Communication Overhead** -- Inter-node reduce-scatter (ZeRO-2) and all-gather (ZeRO-1) latency
5. **MoE All-to-All Routing** -- Intra-node communication for Mixture-of-Experts token routing
6. **PP SendRecv Communication** -- Pipeline parallelism point-to-point overhead: intra-node (NVLink) vs inter-node (EFA), bubble percentage, exposed latency

It supports dense models (Llama 3.1/3.2/3.3, Qwen2.5/3/3.5) and MoE models (DeepSeek-V3/V3.1/V3.2/V4, Qwen3-MoE, Qwen3.5-MoE, Llama-4-Scout/Maverick) on AWS GPU instances (p5/H100, p5en/H200, p6-b200/B200, p6-b300/B300 Ultra, p6e-gb200/GB200 NVL). Results are exported as CSV/JSON and optionally rendered as an AWS-branded HTML report.

## Onboarding

### Step 1: Validate Python 3

Before running the calculator, verify Python 3 is available:

```bash
python3 --version
```

No additional Python packages are required -- the calculator uses only the standard library.

### Step 2: Clone the calculator engine

Clone the calculator repository into the project's working directory:

```bash
git clone https://github.com/paragao/ai-model-calculator.git /tmp/ai-model-calculator
```

This provides the calculator engine (`model_calculations.py` + `utils/`). The `configuration/` directory contains default configs that will be overwritten based on user input using templates from the steering files.

If the repo is already cloned, skip this step. Verify with:

```bash
ls /tmp/ai-model-calculator/model_calculations.py
```

### Step 3: Verify directory structure

After cloning, confirm the following structure exists:

```
/tmp/ai-model-calculator/
  model_calculations.py       # Main orchestrator (do not modify)
  configuration/
    __init__.py
    variants_config.py        # Model definitions (overwritten per analysis)
    hardware_config.py        # Hardware definitions (overwritten per analysis)
    project_config.py         # Training parameters (overwritten per analysis)
    advanced_config.py        # Advanced settings (rarely modified)
  utils/
    __init__.py
    validation.py
    core_calculations.py
    formatting_utils.py
    phase1_memory.py
    phase2_batch.py
    phase3_training.py
    phase4_zero2_comm.py
    phase4_1_zero1_comm.py
    phase5_alltoall_comm.py
    phase6_pp_comm.py
    calculate_variant_fields.py
    calculate_qwen_params.py
```

**CRITICAL**: Never modify files in `utils/` or `model_calculations.py`. Only modify files in `configuration/`.

## When to load steering files

- **Selecting models or customizing architecture** (dense or MoE, pre-defined or custom parameters) -> `model-catalog.md`
- **Selecting hardware platforms** (p5, p5en, custom GPU clusters, node counts) -> `hardware-catalog.md`
- **Configuring training parameters** (parallelism, precision, tokens, MFU, batch size targets) -> `training-config.md`
- **Running the analysis** (the full Step 1-6 workflow, calculation formulas, rules, output format) -> `calculator-workflow.md`
- **Generating an HTML report from results** (aws-branded-report detection, chart specs, section mapping) -> `report-integration.md`
