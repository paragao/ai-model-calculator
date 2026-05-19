"""
Project-specific configuration variables.

Edit these settings for each new analysis. These are the primary parameters
that should be adjusted based on the model architecture and training goals.
"""

# Model architecture parameters
VOCAB = 151_936      # Vocabulary size (Qwen3 default; Llama=128256, Qwen2.5=152064, DeepSeek=129280)
SEQ_LEN = 4096       # Sequence length (context window)
TOPK = 8             # Top-K experts selected per token in MoE routing (0 for dense)
N_EXPERTS = 128      # Total number of experts in MoE layers (0 for dense)

# Parallelism configuration
PP = 1   # Pipeline parallelism degree (layers split across pipeline stages)
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
