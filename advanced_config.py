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
from project_config import PRECISION

# Target batch size configuration
TOKENS_PER_BATCH = 67.1e6  # Chinchilla-optimal batch size for ~160B parameter models

# ZeRO optimization strategies with efficiency multipliers
ZERO_STRATEGY = [
    {"zero": 1, "eff": 1.0},   # ZeRO-1: Optimizer states sharded, best efficiency
    {"zero": 2, "eff": 0.9},   # ZeRO-2: Optimizer + gradients sharded, 10% overhead
    {"zero": 3, "eff": 0.8},   # ZeRO-3: All parameters sharded, 20% overhead
]

# Micro batch sizes to test for memory feasibility
MICROS = [1, 2, 4, 8]

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
NCCL_MEM_BUF = 4.0  # NCCL communication buffer size in GB

# Memory utilization
GPU_MEM_UTILIZATION_THRESHOLD = 0.92  # Use up to 92% of GPU memory

# Training step boundaries (for assessing batch size quality)
LOWER_BOUND_OPTIM_RANGE = 300_000    # Lower bound for "reasonable" training steps
UPPER_BOUND_OPTIM_RANGE = 800_000    # Upper bound for "reasonable" training steps
LOWER_BOUND_LOW_RANGE = 200_000      # Lower bound for "borderline low" steps
UPPER_BOUND_LOW_RANGE = 300_000      # Upper bound for "borderline low" steps
LOWER_BOUND_HIGH_RANGE = 800_000     # Lower bound for "borderline high" steps
UPPER_BOUND_HIGH_RANGE = 1_200_000   # Upper bound for "borderline high" steps
MIN_STEPS = 200_000                  # Minimum acceptable training steps
MAX_STEPS = 1_200_000                # Maximum acceptable training steps

# Batch size constraints
MAX_TOKENS_PER_BATCH = 600e6  # Maximum tokens per batch (600M)
