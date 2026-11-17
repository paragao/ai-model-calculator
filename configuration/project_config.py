"""
Project-specific configuration variables.

Edit these settings for each new analysis. These are the primary parameters
that should be adjusted based on the model architecture and training goals.
"""

# Model architecture parameters
VOCAB = 151_936      # Vocabulary size (Qwen3 default; Llama=128256, Qwen2.5=152064, DeepSeek=129280)
SEQ_LEN = 4096       # Sequence length (context window)
TOPK = 8             # FALLBACK only: a variant's own "topk" wins (see resolve_topk).
                     # Correct for the Qwen3/Qwen3.5-A3B/A10B and DeepSeek-V3.x
                     # entries; Llama-4 Scout and Maverick route to 1 and
                     # Qwen3.5-397B-A17B to 10. DeepSeek-V4 declares none and so
                     # uses this value, which its own params imply is right.
N_EXPERTS = 128      # FALLBACK only: a variant's own "num_experts" wins (see
                     # resolve_experts_per_gpu). Llama-4-Scout has 16, Qwen3.5-397B
                     # 512 and the DeepSeek entries 256.

# Parallelism configuration
PP = 8   # Pipeline parallelism degree (layers split across pipeline stages)
VP = 4   # Virtual pipeline parallelism degree (interleaved schedule; 1 = no VP)
TP = 1   # Tensor parallelism degree (model split within layers)
CP = 1   # Context parallelism degree (sequence/context split across GPUs)
EP = 8   # Expert parallelism degree (experts distributed across GPUs; 1 for dense)

# Training configuration
TOTAL_TOKENS = 15e12  # Total tokens to train on (15T = 15 trillion)
PRECISION = "BF16"    # Training precision: "BF16" (16-bit) or "FP8" (8-bit)
# Memory mode overrides (bytes per parameter); None keeps the mixed-precision accounting of advanced_config. Set by the Trainium
# "bf16_sr" mode (BF16 with stochastic rounding, no FP32 master copy), see lambda_function.NEURON_MEMORY_MODES.
GRAD_BYTES_PER_PARAM_OVERRIDE = None
OPTIM_BYTES_PER_PARAM_OVERRIDE = None

# Performance and time estimation
MFU = 0.40  # Model FLOPs Utilization - fraction of peak achieved (typically 0.35-0.50)
TIME_ESTIMATE_LOW_MULTIPLIER = 0.75   # Optimistic time estimate multiplier
TIME_ESTIMATE_HIGH_MULTIPLIER = 1.25  # Pessimistic time estimate multiplier

# Batch size matching tolerance
MARGIN = 0.02  # Tolerance for matching target batch sizes (2% = 0.02)
# NOTE: MARGIN now has NO code consumer. Its only use was the priority-0
# "tokens_per_batch matches TOKENS_PER_BATCH within MARGIN" check in Phase 2, which
# was unreachable (0 of 14,080 configurations) and was removed when batch
# assessment became relative -- see phase2_batch.classify_step_outliers. Left in
# place rather than deleted because removing a public config constant is a separate
# decision; widening it would not revive the old check, since an exact
# dp*micro*accum*seq_len product still cannot be made to land on an arbitrary
# absolute target.

# Derived values (automatically calculated)
EXPERTS_PER_GPU = N_EXPERTS // EP if EP > 0 else 0  # Number of experts per GPU
