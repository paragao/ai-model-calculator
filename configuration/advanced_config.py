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
TOKENS_PER_BATCH = 4e6  # Target tokens per batch (4M default)

# ZeRO optimization strategies with efficiency multipliers
#
# Communication for all three stages is now modelled: phase 4 covers ZeRO-2
# (gradient reduce-scatter), phase 4.1 covers ZeRO-1 (parameter all-gather), and
# phase 4.2 covers ZeRO-3 (two parameter all-gathers plus the gradient
# reduce-scatter). Per the ZeRO paper (arXiv:1910.02054v3 sec 7.2.1-7.2.2),
# ZeRO-1 and ZeRO-2 each total 2*Psi per step -- the same as baseline DP -- while
# ZeRO-3 totals 3*Psi, a fixed 1.5x.
#
# The `eff` multipliers below are EMPIRICAL and are not derived from those
# volumes. They are a separate, coarser knob applied to training time in phase 3;
# the per-stage volumes in phases 4/4.1/4.2 are the physics. Note the ordering
# below is consistent with the paper (ZeRO-3 costs more than ZeRO-1/2), but the
# specific 0.9 and 0.8 values are uncalibrated.
ZERO_STRATEGY = [
    {"zero": 1, "eff": 1.0},   # ZeRO-1: Optimizer states sharded, best efficiency
    {"zero": 2, "eff": 0.9},   # ZeRO-2: Optimizer + gradients sharded, 10% overhead
    {"zero": 3, "eff": 0.8},   # ZeRO-3: All parameters sharded, 20% overhead
]

# --- ZeRO-3 transient all-gather buffer -------------------------------------
#
# ZeRO-3's resident parameter memory is Psi/Nd, but a layer cannot EXECUTE while
# sharded: DeepSpeed all-gathers each module's full parameters into a transient
# buffer just before it runs, and frees them after. That buffer is real peak
# memory the paper's resident model does not account for.
#
# Both values below are DeepSpeed's own documented DEFAULTS, in PARAMETER
# ELEMENTS (not bytes), from https://www.deepspeed.ai/docs/config-json/:
#   stage3_max_live_parameters  1e9  "The maximum number of parameters resident
#                                    per GPU before releasing. Smaller values use
#                                    less memory, but perform more communication."
#   stage3_prefetch_bucket_size 5e8  "The size of the fixed buffer for prefetching
#                                    parameters. Smaller values use less memory,
#                                    but can increase stalls due to communication."
#
# They are config knobs, not physics: a run that lowers them trades memory back
# for communication, so the modelled buffer describes the DEFAULT configuration.
ZERO3_MAX_LIVE_PARAMS = 1e9
ZERO3_PREFETCH_BUCKET_PARAMS = 5e8

# Micro batch sizes to test for memory feasibility
MICROS = [1, 2, 4, 8, 16]

# Gradient accumulation steps to evaluate
#
# 1 means no accumulation, which is valid and is often optimal at large DP width
# where the global batch is already big enough without it. It was previously
# absent, so that arm was never evaluated.
GRAD_ACCUM_VALUES = [1, 2, 4, 8]

# Precision-dependent parameter size
if PRECISION == "BF16":
    PARAM_BYTES = 2  # 2 bytes per parameter for BF16
else:
    PARAM_BYTES = 1  # 1 byte per parameter for FP8

# KNOWN LIMITATION -- PRECISION affects PARAM_BYTES ONLY, never throughput.
# phase3_training reads hw["peak_tflops_bf16"] unconditionally and no hardware
# preset defines peak_tflops_fp8, so setting PRECISION="FP8" halves parameter and
# activation memory while still computing training time at BF16 throughput. FP8 is
# roughly 2x BF16 on H100/H200/B200/B300/GB200, L4 and L40S, so FP8 runs
# OVERSTATE training time by about 2x. (A10G is Ampere and has no FP8 at all, so
# FP8 on g5 is not a valid configuration.) Fixing this needs a per-preset
# peak_tflops_fp8 field plus precision-aware selection in phase3, which belongs
# with the hardware presets rather than here.

# Model architecture constants
LAYER_NORM = 2  # RMSNorm vectors of width d per layer: input + post-attention

# The 3 SwiGLU FFN matrices: gate_proj, up_proj, down_proj. Used as
# `3 * d * dense_ffn`. NOT the QKV projections -- those are counted separately in
# each variant's attn_per_layer.
FFN_WEIGHT_MATRICES = 3

# Input embedding table + output LM head. NOT "word + position" embeddings: every
# model in the catalog uses RoPE and has no learned positional table. Reduced to
# 1 for a variant with "tied_embeddings": True, where the LM head IS the input
# embedding matrix (Llama-3.2-3B is the only such variant).
NUM_EMBEDDINGS_TABLES = 2

# Optimizer memory footprint
#
# Mixed-precision Adam keeps its state in FP32 REGARDLESS of parameter precision:
#   fp32 master weights 4 + fp32 momentum 4 + fp32 variance 4 = 12 bytes/param.
#
# OPTIM_BYTES is consumed as a MULTIPLIER ON PARAMETER MEMORY, not as bytes per
# parameter -- see calculate_memory_breakdown: `model_mem_gb * optim_bytes / dp`,
# where model_mem_gb already carries PARAM_BYTES. So the effective footprint is
# PARAM_BYTES * OPTIM_BYTES bytes/param, and the multiplier must be derived from
# PARAM_BYTES to keep that product at 12.
#
# A fixed 6 was correct only in BF16 (2 * 6 = 12). Under FP8 it gave 1 * 6 = 6,
# understating optimizer memory 2x -- the optimizer does not shrink just because
# the parameters are fp8.
OPTIM_BYTES_PER_PARAM = 12  # fp32 master + fp32 momentum + fp32 variance
OPTIM_BYTES = OPTIM_BYTES_PER_PARAM / PARAM_BYTES  # BF16 -> 6.0, FP8 -> 12.0

# Activation memory estimation (empirical multipliers)
EMPIRICAL_ACT_MULTIPLIER = 12  # Multiplier for activation memory per layer
SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER = 0.45  # Reduction from selective checkpointing

# Communication buffer configuration
FWD_BWD_ROUTING_BUFF_PASSES = 2  # All-to-all buffer passes for MoE routing
NCCL_MEM_BUF = 6.0  # NCCL base communication buffer size in GB (includes CUDA context)
NCCL_EP_SCALING_FACTOR = 1.0  # Additional GB per EP rank beyond 1 (all-to-all workspace)

# Training-time overhead modeling (Phase 3)
# Fraction of MoE all-to-all communication that is NOT overlapped with compute
# and therefore adds to wall-clock time per step. Empirical; 0.0 = fully
# overlapped (no exposed a2a cost), 1.0 = fully exposed (upper bound).
A2A_EXPOSED_FRACTION = 0.5

# Attention FLOPs factor for the O(seq_len^2) attention term in training time.
# The 6ND compute approximation captures only the linear-in-tokens matmuls
# (QKV/O projections, FFN, experts); it omits the attention score (QK^T) and
# context (softmax*V) batched matmuls, which scale as O(seq_len^2 * d) per
# layer. This factor accounts for both attention matmuls across forward+backward
# (forward ~4 = 2 matmuls * 2 FLOPs/MAC, x3 for fwd+bwd = 12). Set to 0 to
# disable the attention term (reverts to pure 6ND).
ATTENTION_FLOPS_FACTOR = 12

# PyTorch memory allocator fragmentation
# Calibrated against nsys profiling of Qwen3-235B-A22B on 64 B300 GPUs (PP=8, EP=8):
# actual measured 150 GB vs raw calculated ~121 GB → ratio ≈ 1.24x.
# Accounts for CUDA caching allocator fragmentation, memory alignment, and
# small transient buffers not explicitly modeled.
#
# INTERACTION with ZERO3_MAX_LIVE_PARAMS above -- RESOLVED, no double-count.
# This factor multiplies the total INCLUDING the explicit ZeRO-3 transient
# all-gather buffer, which would double-count if the calibration run had itself
# been a ZeRO-3 run. It was not:
#   1. utils/validate_moe_memory.py, the harness that reproduces the measured
#      150 GB, pins zero_strategy=1 explicitly.
#   2. That config is PP=8, EP=8, TP=1 on 64 GPUs, so DP = 64/(8*1*1*8) = 1.
#      At DP=1 every ZeRO /dp divisor is a no-op and ZeRO-1 and ZeRO-2 are
#      byte-identical (124.2 GB raw for both), so no stage shards anything and a
#      real run at that width would not be gathering parameters either.
#   3. zero3_transient is therefore 0.00 GB for the calibration config, so the
#      1.24x was fitted to a raw total containing NONE of that buffer.
# For scale, had the calibration been ZeRO-3 the buffer would have contributed
# 2.00 GB of a 124.2 GB raw total, i.e. 1.6% -- so even the worst case was small.
FRAGMENTATION_FACTOR = 1.24  # 1.15-1.25x; empirical overhead from caching allocator

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

# The step bounds above are CONTIGUOUS and are consumed as half-open intervals by
# phase2_batch.assess_training_steps, so every step count is classified. They were
# once all tested strictly on both sides, which left 200_000 / 300_000 / 800_000
# matching no branch and reported as "unknown".
#
# UNRESOLVED -- these bounds DISAGREE with TOKENS_PER_BATCH in project_config:
#   TOTAL_TOKENS / TOKENS_PER_BATCH = 15e12 / 4e6 = 3,750,000 steps, which is 3.1x
#   MAX_STEPS. The optimal band instead implies a batch of
#   TOTAL_TOKENS/UPPER_BOUND_OPTIM_RANGE .. TOTAL_TOKENS/LOWER_BOUND_OPTIM_RANGE
#   = 18.8M .. 50.0M tokens, i.e. 5x-12x the configured target.
#   Worse, neither target window is reachable: tokens_per_batch is
#   dp*micro*accum*seq_len and cannot land within +/-MARGIN (2%) of an arbitrary
#   absolute number, so priority 0 is never returned at all -- measured 0 hits
#   across 14,080 swept configs.
# Fixing it means choosing an intent (a 4M batch, or 300k-800k steps), so it is
# deliberately not guessed. See the note in assess_training_steps.

# Batch size constraints
#
# AUTO-COMPUTED -- see compute_max_tokens_per_batch below. The value here is only
# the default at the reference configuration, kept so existing callers that import
# the name keep working. Prefer the function, which takes the actual dp and
# seq_len.
#
# PROVENANCE of the old hard-coded 600e6: none that could be traced. It is
# inconsistent with this module's own step bounds by 8x --
# TOTAL_TOKENS / 600e6 = 25,000 steps, which is 8.0x BELOW MIN_STEPS (200,000).
# That is why it never fired: measured 0 hits across 14,080 swept configs. A cap
# that admits batches implying 25k steps is not a cap at all when the same module
# calls anything under 200k steps "too few steps".
MAX_TOKENS_PER_BATCH = 600e6  # Maximum tokens per batch (600M)


def compute_max_tokens_per_batch(total_tokens, dp, seq_len, min_steps=None):
    """Largest sensible global batch, derived rather than hard-coded.

    Two independent bounds, and the batch must respect both:

    1. STEP BUDGET (statistical). steps = total_tokens / batch, so requiring at
       least min_steps optimizer updates caps the batch at total_tokens/min_steps.
       This is the binding bound in practice.
    2. PARALLELISM FLOOR (hardware). The smallest batch the topology can even
       express is dp * seq_len -- one sequence per data-parallel rank, micro=1,
       accum=1. No cap below this is satisfiable, so it wins when the two cross,
       and the caller should treat that as "this parallelism cannot hit the step
       budget" rather than as a usable cap.

    Returned value is max(...) of the two: the step ceiling normally, raised to
    the floor when the topology is too wide to reach min_steps.

    WHY THIS SHAPE, and what the literature does and does not settle
    ----------------------------------------------------------------
    The governing quantity in the literature is the CRITICAL BATCH SIZE -- the
    point past which a larger batch stops buying proportionally fewer steps.
    Zhang et al., "How Does Critical Batch Size Scale in Pre-training?"
    (arXiv:2410.21676, Harvard/Amazon) decouple its drivers and find CBS
    "scales primarily with data size rather than model size", fitting
    B* = 22.91 * D^0.47 (D in billions, at context length 512) and
    B* = 621.341 * N^0.087 -- "barely invariant to model size". So the right
    independent variable IS the dataset size, which is what this function uses.

    Its absolute constant does NOT transfer to frontier scale, so it is not used
    here. Extrapolated to D = 15T that law gives 1.08M tokens, which is 15x to
    58x BELOW what the published frontier runs actually use. It was fitted on
    85M-1.2B parameter models over <=120B tokens at context 512, two orders of
    magnitude short of a 15T-token run. Forcing B = k * D^0.47 through the
    published runs gives k = 112,351 (Llama 2 70B), 171,140 (Llama 3 405B) and
    689,808 (DeepSeek-V3) -- a ~6x spread -- so any single k would be invented
    rather than derived. The exponent may well be right; the constant is not
    available, and closing that gap needs frontier-scale CBS measurements that
    nobody has published.

    So this function anchors on the step budget, which is the user's own declared
    intent, and the published runs corroborate that band rather than supply a
    constant to it. At TOTAL_TOKENS = 15e12 the [MIN_STEPS, MAX_STEPS] window of
    [200k, 1.2M] implies a batch of 12.5M to 75.0M tokens, and BOTH frontier runs
    land inside it:

      Llama 3 405B  final batch 16.0M tokens  (arXiv:2407.21783 S3.4.1)
      DeepSeek-V3   final batch 62.9M tokens  (arXiv:2412.19437 S4.2,
                                               15360 sequences x 4096)

    Note both runs RAMP the batch and neither holds it constant, which is what
    reconciles TOKENS_PER_BATCH = 4e6 with MAX_STEPS = 1.2e6: 4M is Llama 3's
    INITIAL batch, not its steady-state one. Its schedule 4M -> 8M (after 252M
    tokens) -> 16M (after 2.87T) reconstructs to 1,154,406 steps against the
    paper's "over 1,200,000", a 3.8% agreement, at a time-averaged 13.5M tokens
    per step. This engine models a SINGLE fixed batch, so it cannot express a
    ramp; the band is the honest single-value stand-in for one.

    Args:
        total_tokens: Training token budget (TOTAL_TOKENS).
        dp: Data-parallel degree of the configuration being checked.
        seq_len: Sequence length in tokens.
        min_steps: Minimum acceptable optimizer steps; defaults to MIN_STEPS.

    Returns:
        Maximum tokens per batch, in tokens.
    """
    if min_steps is None:
        min_steps = MIN_STEPS
    if min_steps <= 0:
        raise ValueError(f"min_steps must be positive, got {min_steps}")
    if dp <= 0 or seq_len <= 0:
        raise ValueError(f"dp and seq_len must be positive, got dp={dp}, seq_len={seq_len}")

    step_ceiling = total_tokens / min_steps
    parallelism_floor = dp * seq_len
    return max(step_ceiling, parallelism_floor)


def describe_batch_budget(total_tokens, dp, seq_len,
                          min_steps=None, max_steps=None):
    """The full feasible batch band for a configuration, for reporting.

    Returns a dict with the step-budget band, the parallelism floor, the
    resulting cap, and whether the topology can reach the step budget at all.
    """
    if min_steps is None:
        min_steps = MIN_STEPS
    if max_steps is None:
        max_steps = MAX_STEPS

    floor = dp * seq_len
    band_low = total_tokens / max_steps
    band_high = total_tokens / min_steps
    return {
        "parallelism_floor": floor,
        "band_low": band_low,
        "band_high": band_high,
        "max_tokens_per_batch": max(band_high, floor),
        # When the smallest expressible batch already exceeds the step ceiling,
        # no configuration on this topology can reach min_steps.
        "floor_exceeds_band": floor > band_high,
        "implied_steps_at_floor": total_tokens / floor,
    }

# Hardware memory specifications (GB)
#
# REFERENCE VALUES ONLY -- the calculation always uses the hardware entry's
# own mem_gb. Six of these eight are referenced nowhere outside this file, so
# editing one does NOT change any result. All verified against AWS specs.
H100_MEM_GB = 80    # H100 GPU memory capacity
H200_MEM_GB = 141   # H200 GPU memory capacity
B200_MEM_GB = 180   # B200 GPU memory capacity
B300_MEM_GB = 268   # B300 Ultra: 268 GB USABLE of 288 raw; AWS states 2.1 TB / 8 GPUs
GB200_MEM_GB = 186  # GB200 NVL GPU memory: AWS states 13.4 TB HBM3e / 72 GPUs
A10G_MEM_GB = 24    # A10G GPU memory capacity
L4_MEM_GB = 24      # L4 GPU memory capacity
L40S_MEM_GB = 48    # L40S GPU memory capacity
