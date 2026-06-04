# Training Configuration

## Training parameters

These have sensible defaults but the user can override them. Show the current values and ask if anything needs changing.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VOCAB` | `128,256` | Vocabulary size (adjust per model family) |
| `SEQ_LEN` | `4,096` | Sequence length (context window) |
| `N_EXPERTS` | `128` | Total experts (MoE only; set 0 for dense) |
| `TOPK` | `8` | Top-K expert routing (MoE only; set 0 for dense) |
| `PP` | `1` | Pipeline parallelism degree |
| `VP` | `1` | Virtual pipeline parallelism degree (interleaved schedule; 1 = no VP). Typical: VP=4 for PP=8, VP=2 for PP=4. |
| `TP` | `1` | Tensor parallelism degree |
| `CP` | `1` | Context parallelism degree |
| `EP` | `8` | Expert parallelism degree (set 1 for dense) |
| `TOTAL_TOKENS` | `15e12` | Total training tokens |
| `PRECISION` | `"BF16"` | Precision: "BF16" or "FP8" |
| `MFU` | `0.40` | Model FLOPs Utilization target (0.35-0.50 typical) |
| `TOKENS_PER_BATCH` | `4e6` | Target tokens per batch |

## Smart defaults by model type

Apply these automatically based on the selected model -- do not ask the user:

| Model Family | VOCAB | N_EXPERTS | TOPK | EP |
|-------------|-------|-----------|------|-----|
| Llama 3.1 (dense) | 128,256 | 0 | 0 | 1 |
| Qwen3 dense | 151,936 | 0 | 0 | 1 |
| Qwen3 MoE | 151,936 | 128 | 8 | 8 |
| Mixtral | 32,000 | 8 | 2 | 8 |
| DeepSeek-V3 | 129,280 | 256 | 8 | 8 |

For all dense models: set `EP=1`, `N_EXPERTS=0`, `TOPK=0`.
For all MoE models: set EP to a divisor of N_EXPERTS (default 8).

## The project_config.py template

Save this file to `/tmp/ai-model-calculator/configuration/project_config.py` before running. Update values based on user selections and smart defaults.

```python
"""
Project-specific configuration variables.

Edit these settings for each new analysis. These are the primary parameters
that should be adjusted based on the model architecture and training goals.
"""

# Model architecture parameters
VOCAB = 128_256  # Vocabulary size (Llama 3.1 default; Qwen3=151936, Mixtral=32000)
SEQ_LEN = 4096   # Sequence length (context window)
TOPK = 8         # Top-K experts selected per token in MoE routing (0 for dense)
N_EXPERTS = 128  # Total number of experts in MoE layers (0 for dense)

# Parallelism configuration
PP = 1   # Pipeline parallelism degree (layers split across pipeline stages)
VP = 1   # Virtual pipeline parallelism degree (interleaved 1F1B; 1 = no VP)
TP = 1   # Tensor parallelism degree (model split within layers)
CP = 1   # Context parallelism degree (sequence/context split across GPUs)
EP = 8   # Expert parallelism degree (experts distributed across GPUs; 1 for dense)

# Training configuration
TOTAL_TOKENS = 15e12  # Total tokens to train on (15T = 15 trillion)
PRECISION = "BF16"    # Training precision: "BF16" (16-bit) or "FP8" (8-bit)

# Performance and time estimation
MFU = 0.40  # Model FLOPs Utilization - fraction of peak achieved (typically 0.35-0.50)
TIME_ESTIMATE_LOW_MULTIPLIER = 0.75   # Optimistic time estimate multiplier
TIME_ESTIMATE_HIGH_MULTIPLIER = 1.25  # Pessimistic time estimate multiplier

# Batch size matching tolerance
MARGIN = 0.02  # Tolerance for matching target batch sizes (2% = 0.02)

# Derived values (automatically calculated)
EXPERTS_PER_GPU = N_EXPERTS // EP if EP > 0 else 0  # Number of experts per GPU
```

## The advanced_config.py template

Save this file to `/tmp/ai-model-calculator/configuration/advanced_config.py`. Rarely needs modification -- only change `TOKENS_PER_BATCH` if the user specifies a target batch size.

```python
"""
Advanced configuration variables.

These are expert-level settings that rarely need modification.
Only change these if you have deep understanding of:
- Memory optimization strategies (ZeRO)
- Distributed training efficiency
- Activation checkpointing
- Training dynamics and convergence

Most users should leave these at their default values.
"""

# Import PRECISION from project config to determine PARAM_BYTES
from .project_config import PRECISION

# Target batch size configuration
TOKENS_PER_BATCH = 4e6  # Target tokens per batch (4M -- adjust per model size)

# ZeRO optimization strategies with efficiency multipliers
ZERO_STRATEGY = [
    {"zero": 1, "eff": 1.0},   # ZeRO-1: Optimizer states sharded, best efficiency
    {"zero": 2, "eff": 0.9},   # ZeRO-2: Optimizer + gradients sharded, 10% overhead
    {"zero": 3, "eff": 0.8},   # ZeRO-3: All parameters sharded, 20% overhead
]

# Micro batch sizes to test for memory feasibility
MICROS = [1, 2, 4, 8, 16]

# Gradient accumulation steps to evaluate
GRAD_ACCUM_VALUES = [4, 8, 16, 32, 64, 128]

# Precision-dependent parameter size
if PRECISION == "BF16":
    PARAM_BYTES = 2  # 2 bytes per parameter for BF16
else:
    PARAM_BYTES = 1  # 1 byte per parameter for FP8

# Model architecture constants
LAYER_NORM = 2  # Pre- and post-attention layer norms
FFN_WEIGHT_MATRICES = 3  # Query, key, value projections + FFN matrices
NUM_EMBEDDINGS_TABLES = 2  # Word embeddings + position embeddings

# Optimizer memory footprint
OPTIM_BYTES = 6  # 2B momentum + 2B variance + 2B FP32 master weights

# Activation memory estimation (empirical multipliers)
EMPIRICAL_ACT_MULTIPLIER = 12  # Multiplier for activation memory per layer
SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER = 0.45  # Reduction from selective checkpointing

# Communication buffer configuration
FWD_BWD_ROUTING_BUFF_PASSES = 2  # All-to-all buffer passes for MoE routing
NCCL_MEM_BUF = 6.0  # NCCL base communication buffer size in GB (includes CUDA context)
NCCL_EP_SCALING_FACTOR = 1.0  # Additional GB per EP rank beyond 1 (all-to-all workspace)

# PyTorch memory allocator fragmentation
FRAGMENTATION_FACTOR = 1.24  # Empirical overhead from caching allocator (calibrated on Qwen3-235B)

# Memory utilization
GPU_MEM_UTILIZATION_THRESHOLD = 0.92  # Use up to 92% of GPU memory

# Training step boundaries (for assessing batch size quality)
LOWER_BOUND_OPTIM_RANGE = 300_000
UPPER_BOUND_OPTIM_RANGE = 800_000
LOWER_BOUND_LOW_RANGE = 200_000
UPPER_BOUND_LOW_RANGE = 300_000
LOWER_BOUND_HIGH_RANGE = 800_000
UPPER_BOUND_HIGH_RANGE = 1_200_000
MIN_STEPS = 200_000
MAX_STEPS = 1_200_000

# Batch size constraints
MAX_TOKENS_PER_BATCH = 600e6  # Maximum tokens per batch (600M)

# Hardware memory specifications (GB)
H200_MEM_GB = 141  # H200 GPU memory capacity
H100_MEM_GB = 80   # H100 GPU memory capacity
B200_MEM_GB = 180  # B200 GPU memory capacity
B300_MEM_GB = 268  # B300 Ultra GPU memory capacity
GB200_MEM_GB = 185 # GB200 NVL GPU memory capacity
```
