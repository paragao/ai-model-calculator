# AI Model Training Calculator

A [Kiro Power](https://kiro.dev/docs/powers/) that analyzes GPU memory, batch configuration, training time, and communication overhead for LLM training across AWS hardware platforms.

## Getting Started

### Step 1: Open the Powers panel in Kiro

In your Kiro IDE, open the command palette (`Cmd+Shift+P` on macOS or `Ctrl+Shift+P` on Linux) and search for:

```
Kiro: Add Power
```

Select the command to open the power installation dialog.

### Step 2: Enter the repository URL

When prompted for the Git repository URL, paste:

```
https://github.com/paragao/ai-model-calc-power.git
```

Press Enter. Kiro will clone the power into your workspace under `.kiro/powers/ai-model-calc/`.

### Step 3: Verify the installation

After installation, confirm the power appears in your workspace:

```
.kiro/
  powers/
    ai-model-calc/
      POWER.md
      steering/
        calculator-workflow.md
        model-catalog.md
        hardware-catalog.md
        training-config.md
        report-integration.md
```

You can also check the Kiro Powers panel in the sidebar — the "AI Model Training Calculator" should be listed as an installed power.

### Step 4: Confirm Python 3 is available

The calculator uses only the Python standard library, so no additional packages are needed. Just verify Python 3.8+ is on your path:

```bash
python3 --version
```

### Step 5: Start using the power

Once installed, Kiro automatically loads the power's steering files when you ask about training infrastructure. Open a chat session and try prompts like:

- "Calculate memory requirements for training Llama 3.1 70B on p5en instances"
- "How many H200 GPUs do I need to train a 13B model in 2 weeks?"
- "Compare training configurations for DeepSeek-V3 on p5 vs p5en"
- "What's the minimum hardware to fit Qwen3-235B for inference?"

Kiro will walk you through selecting a mode, model, and hardware, then run the 5-phase analysis and produce CSV/JSON results with an optional HTML report.

### Optional: Install the AWS Branded Report power

For automatic HTML report generation after each analysis, install the companion power using the same "Add Power" command with:

```
https://github.com/paragao/aws-branded-report-power.git
```

When both powers are installed, the calculator will automatically generate a professional AWS-branded HTML report from the analysis results.

## Modes of Operation

The calculator supports three modes, each answering a different infrastructure planning question:

### Mode A — "How many instances do I need to finish in time X?"

You provide:
- Model to train
- Dataset size (in tokens)
- Target training time (e.g., "2 months", "30 days")

The calculator sweeps multiple node counts and finds the minimum cluster size that meets your time target.

### Mode B — "How long will training take?"

You provide:
- Model to train
- Dataset size (in tokens)
- Number of instances and instance type

The calculator estimates wall-clock training duration with low/high confidence ranges.

### Mode C — "What's the minimum to fit this model?"

You provide:
- Model to train (or serve/fine-tune)
- Instance type

The calculator determines the minimum number of instances required to hold the model in GPU memory. This is a pure memory-fit calculation — no dataset size needed. Results are presented for inference, fine-tuning, and full pre-training scenarios side by side.

## What it calculates

| Phase | Output |
|-------|--------|
| 1. Memory Analysis | Per-GPU memory breakdown (model, gradient, optimizer, activation) across ZeRO stages |
| 2. Batch Configuration | Optimal micro batch size and gradient accumulation steps |
| 3. Training Time | Wall-clock estimates with confidence ranges |
| 4. Communication Overhead | ZeRO reduce-scatter and all-gather latency |
| 5. MoE Routing | All-to-all communication for Mixture-of-Experts models |

Supports dense models (Llama, Qwen) and MoE models (Mixtral, DeepSeek-V3, Qwen3-MoE) on AWS GPU instances (p5/H100, p5en/H200).

## Repository structure

```
POWER.md                    # Power manifest and onboarding instructions
steering/
  calculator-workflow.md    # Full calculation workflow and formulas
  model-catalog.md          # Supported model architectures
  hardware-catalog.md       # AWS GPU instance definitions
  training-config.md        # Parallelism, precision, and batch settings
  report-integration.md    # Optional HTML report generation
```
