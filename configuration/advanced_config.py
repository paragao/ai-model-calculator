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
#
# NOT a target any longer -- no code tries to match it. Phase 2's "does this batch
# hit TOKENS_PER_BATCH" check was unreachable and was removed. The value is Llama 3
# 405B's INITIAL batch (arXiv:2407.21783 S3.4.1: "an initial batch size of 4M
# tokens and sequences of length 4,096"), which that run doubled to 8M and then 16M.
#
# Its two live consumers read it as exactly that -- the smallest batch in published
# practice, not a goal:
#   - compute_absurd_step_ceiling: total_tokens / this = the step count above which
#     a configuration asks for a batch no published frontier run ever used.
#   - Phase 6's global-batch approximation.
TOKENS_PER_BATCH = 4e6  # Llama 3 405B ramp-START batch, used as a credibility floor

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

# PARAM_STORAGE_BYTES is what the weights actually OCCUPY, which is not PARAM_BYTES.
#
# PARAM_BYTES is a COMPUTE precision. On this stack FP8 is requested with
# `--compute_dtype fp8_cs` (modified-files/launch.sh in the dgxc-benchmarks-efa
# repo), and that is the only precision flag the recipe passes -- there is no
# separate storage or params dtype. Transformer Engine therefore keeps the
# parameters resident in BF16 and produces cached FP8 cast copies per GEMM, each with
# ONE FP32 per-tensor scale factor -- `fp8_cs` is current scaling, so there is no
# amax history buffer. FP8 ADDS resident bytes on top of the BF16 weights; it does
# not replace them.
#
# MEASURED, Llama 70B LoRA finetune on H200 p5en, 32 GPUs, TP=4 PP=2 VP=20 MBS=1,
# identical cluster/container/parallelism with precision as the ONLY difference
# (jobs 1236 bf16 and 1238 fp8). NOTE THE REGIME: this is a LoRA run and this
# calculator models full pretraining, which measurably differ -- see "CONSEQUENCE FOR
# VALIDATION" beside ACT_BYTES_PER_ELEMENT before using this pair as a target:
#     bf16  37.8 GB allocated / 38.5 reserved
#     fp8   41.1 GB allocated / 41.3 reserved   = +8.73% allocated
# FP8 cost MORE memory, so halving parameter memory was wrong in SIGN. Holding
# parameters at BF16 moves the predicted FP8/BF16 total ratio from 0.9028 to
# 1.0000 against a measured 1.0873, cutting the error from -17.0% to -8.0%.
#
# The residual +8.7% is the FP8 weight cast copies. It is NOT scaling metadata:
# `fp8_cs` is CURRENT scaling, one FP32 scale per tensor with no amax history, which
# is negligible. The cast-copy term is now measured but deliberately not applied --
# see the measured table and the 2.67x scale discrepancy beside
# ACT_BYTES_PER_ELEMENT, and open question 3 above.
PARAM_STORAGE_BYTES = 2  # weights are resident in BF16 at either PRECISION

# PRECISION no longer affects any memory term. It selects the GEMM precision, and
# through that the throughput peak; parameters, gradients, optimizer state and
# activations are all resident in BF16 at either setting, each for its own measured
# reason recorded below. A sweep of all 704 variant x hardware pairs is byte-identical
# between PRECISION="BF16" and PRECISION="FP8", and a test pins that.
#
# Throughput is handled: every hardware preset defines peak_tflops_fp8 and
# phase3_training selects on PRECISION, refusing g5 where the value is 0 because
# A10G is Ampere and has no FP8 at all. The earlier note here claiming
# "phase3_training reads peak_tflops_bf16 unconditionally and no hardware preset
# defines peak_tflops_fp8" described a defect that has since been fixed.
#
# Three precision questions, all measured. Two resolved, one open:
#
# 1. GRADIENTS scale with PARAM_BYTES. -- RESOLVED, see GRAD_BYTES below. It is
#    now derived from PARAM_STORAGE_BYTES, so it is a constant 1.0 and gradients
#    sit at GRAD_BYTES_PER_PARAM at either precision. An FP8 run charges 3.846 GB
#    on Llama-3.1-8B, matching BF16, because gradients reduce in BF16 either way.
#    The same correction applies to OPTIM_BYTES.
#
# 2. ACTIVATIONS do NOT scale with PRECISION. -- RESOLVED. Both halves of the
#    activation formula are now precision-invariant: see ACT_BYTES_PER_ELEMENT
#    below, which replaced the PARAM_BYTES factor on the MoE expert intermediate.
#    Decided against MEASUREMENT rather than by symmetry -- the controlled bf16/fp8
#    pair shows FP8 costing 8.73% MORE memory, so making activations cheaper under
#    FP8 moved the prediction further from reality, not closer.
#
# 3. STILL OPEN -- THE FP8 WEIGHT CAST COPY IS NOT MODELLED.
#    This is now the single largest known gap. With parameters, gradients,
#    optimizer state and activations all precision-invariant, this model predicts
#    IDENTICAL memory at BF16 and FP8, so it is LOW for an FP8 run.
#
#    Two earlier claims in this entry were WRONG and are corrected here.
#
#    (a) "Scaling metadata" is NOT part of the gap. The recipe requests FP8 with
#        `--compute_dtype fp8_cs`, which is CURRENT scaling: one FP32 scale factor
#        per tensor, computed on the fly, with NO amax history buffer. That is a
#        handful of bytes per tensor and is negligible. The gap is cast copies.
#
#    (b) "The +8.73% total cannot be decomposed" was wrong. It decomposes cleanly,
#        because the weight cast copy is CONSTANT in token count while the
#        activation delta is LINEAR in it, so a token sweep separates them by
#        regression. Done in jobs 348/350/351 at r2 = 1.000000; see the measured
#        values and their (1 - recompute_fraction) scaling beside
#        ACT_BYTES_PER_ELEMENT below.
#
#    What remains open is not the mechanism but its MAGNITUDE AT SCALE. The measured
#    2.0 bytes/param is exact on a single non-pipelined stack of up to 16 layers and
#    9.9e9 parameters, yet it overshoots the one controlled pair by 2.67x. The
#    untested cause is pipelined residency, and the probe cannot reach it. Still do
#    NOT invent a fraction here: there are now good measurements pointing at a
#    constant that demonstrably does not hold at scale, which is a more persuasive
#    version of the same trap, not an escape from it.
#
# OPTIM_BYTES is correct across precisions -- see its derivation below, which
# divides by PARAM_BYTES to hold the product at 12 bytes/param.

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
# Divided by PARAM_STORAGE_BYTES, not PARAM_BYTES, because model_mem_gb now carries
# the STORAGE size. Since storage is BF16 at either precision this is a constant
# 6.0, and the product stays at OPTIM_BYTES_PER_PARAM as it must.
OPTIM_BYTES = OPTIM_BYTES_PER_PARAM / PARAM_STORAGE_BYTES  # 6.0 at either precision

# GRAD_BYTES is a MULTIPLIER ON PARAMETER MEMORY, the same shape as OPTIM_BYTES
# above, because model_mem_gb already carries PARAM_BYTES.
#
# Gradients are NOT stored at PARAM_BYTES. An FP8 recipe (e.g. Transformer Engine)
# uses FP8 only for the GEMM inputs and reduces/accumulates gradients in BF16 or
# higher, so an FP8 run must still pay 2 bytes/param for gradients. Deriving the
# multiplier from PARAM_BYTES holds the product at GRAD_BYTES_PER_PARAM, exactly as
# OPTIM_BYTES holds its product at 12.
#
# Before this existed, grad_mem was model_mem_gb directly, so FP8 charged 1
# byte/param -- measured on Llama-3.1-8B, 1.923 GB against the correct 3.846.
GRAD_BYTES_PER_PARAM = 2  # BF16 gradient accumulation and reduction
# Divided by PARAM_STORAGE_BYTES for the same reason as OPTIM_BYTES above. Both
# are now constant across PRECISION, which is the point: neither gradients nor
# optimizer state shrink under FP8, and neither do the weights themselves.
GRAD_BYTES = GRAD_BYTES_PER_PARAM / PARAM_STORAGE_BYTES  # 1.0 at either precision

# --- Output-logits memory -------------------------------------------------
#
# The LM head produces a (micro, seq_len, vocab) tensor and the cross-entropy
# backward needs its gradient. vocab is 128,256 to 248,320 across this catalog, so
# this term is large and was previously not modelled at all.
#
# 8 bytes per element covers a bf16 logits tensor plus the fp32 softmax and its
# gradient, which is the standard unfused working set. Checked against the
# component figures: for Llama-3.1-8B at seq_len 4096 and vocab 128,256 this gives
# 4.20 GB at micro=1 and 33.6 GB at micro=8.
LOGITS_BYTES_PER_ELEMENT = 8

# FUSED CROSS-ENTROPY ASSUMPTION -- the dominant knob in this term.
#
# A fused or chunked CE (Liger's fused_linear_cross_entropy, a compiled chunked
# loss) never materialises the full logits tensor. It walks the flattened
# (micro * seq_len) rows in chunks, and for each chunk computes logits, softmax and
# the gradient, then backpropagates straight into the hidden states and the LM head
# weight gradient. Only ONE chunk is live at a time.
#
# The behavioural consequence matters more than the number: above the chunk size
# this term stops scaling with micro entirely, whereas unfused it grows linearly.
#
# LOGITS_CHUNK_TOKENS is an ASSUMPTION, not a measured value. Real implementations
# size the chunk from available memory, so treat it as the knob to set rather than a
# property of the model. Set LOGITS_FUSED_CE = False to charge the full unfused
# tensor instead, which is the right choice for a plain nn.CrossEntropyLoss over a
# materialised logits tensor.
LOGITS_FUSED_CE = True
LOGITS_CHUNK_TOKENS = 1024  # rows whose logits are live at once under fused CE

# Activation memory estimation (empirical multipliers)
EMPIRICAL_ACT_MULTIPLIER = 12  # Multiplier for activation memory per layer

# ACT_BYTES_PER_ELEMENT resolves the disagreement the two halves of the activation
# formula used to have about precision. EMPIRICAL_ACT_MULTIPLIER is already BYTES
# per element of (seq_len * micro * d) -- the expression divides by 1e9 with no
# other byte factor -- so it is an implicitly-BF16 figure and cannot scale. The MoE
# expert intermediate in the SAME expression counted elements and multiplied by
# PARAM_BYTES, so FP8 saved 20.0-40.5% on a MoE layer and exactly 0% on a dense
# one. That split is not physical: a dense FFN intermediate is as FP8-eligible as
# an expert one.
#
# Both halves are now precision-invariant. Chosen against measurement rather than
# by symmetry: the controlled bf16/fp8 pair above shows FP8 costing 8.73% MORE
# memory, so of the candidate treatments, making activations CHEAPER under FP8
# moved the prediction further from reality --
#     both invariant (this)    FP8/BF16 ratio 0.9028   error -17.0%
#     dense x0.75              ratio 0.8528            error -21.6%
#     dense x0.50              ratio 0.8028            error -26.2%
# against the measured 1.0873. Real FP8 activation savings are not zero, but they
# are smaller than the cast-copy overhead that offsets them.
#
# CORRECTION (measured, Slurm jobs 348/349 on H200, TE 2.9.0, Float8CurrentScaling):
# an earlier version of this note claimed the two terms were "not separable from a
# single measured total". That was WRONG -- they separate cleanly, because the weight
# cast copy is CONSTANT in token count and the activation delta is LINEAR in it, so a
# token sweep splits them by regression. Measured at r2 = 1.000000:
#     FP8 weight cast copies  = +1.9998 bytes/param  (rowwise AND columnwise both
#                               resident on Hopper; matches NVIDIA's "same footprint
#                               as BF16" for the default fused cast-transpose)
#     saved-activation delta  = 3.5000 bytes/element SAVED of the 12 = 29.2%
#                               (matches NVIDIA's documented 20-30% empirical saving)
# Job 349 then swept depth L = 1,2,4,8,16: a/param held at 1.9998 with spread 0.0000
# and a(L) was linear through the origin, so bytes-per-param is the right FORM.
#
# Neither constant is shipped, deliberately, and the reason has CHANGED. An earlier
# version of this note said the untested cause was that the probe had no activation
# recomputation while the recipe recomputes ~50% of layers. That has now been
# measured, and it was the right mechanism but it does NOT close the gap.
#
# MEASURED, jobs 350/351 (H200, TE 2.9.0, te.checkpoint, recompute_num_layers=1 per
# block, r2 = 1.000000 at EVERY point, in two independent arms):
#     recomputed fraction f : 0.00    0.25    0.50    0.75    1.00
#     weight copies B/param : 1.9998  1.4998  0.9999  0.5000  0.0001
#     activation saved B/tok: -114688 -86016  -57344  -28672       0
# so BOTH FP8 terms scale by (1 - f) exactly:  a(f) = 2.0*(1-f), s(f) = 3.5*(1-f).
# The mechanism is that a recomputed block keeps neither its activations nor its fp8
# weight copy -- both are rebuilt during the recompute pass. Plumbing was proven in
# the same job: retained forward memory fell 128.19 -> 8.00 MiB (93.8%) with
# te.checkpoint, which TE reports as FP8-aware.
#
# MEASURED, job 351 ARM B -- a FROZEN base (n_trainable = 0, modelling LoRA):
#     weight copies B/param : 1.9998 at f=0, IDENTICAL to the trainable arm
#     activation saved B/tok: 0.00 at every f
# Freezing drops NEITHER fp8 weight copy, so "LoRA keeps only the rowwise copy" is
# refuted. But the fp8 activation saving vanishes entirely, and cleanly: a frozen
# weight needs no wgrad, wgrad is what consumes the saved GEMM input, so that input
# is not saved in EITHER precision and there is no 2->1 byte saving to collect.
#
# CONSEQUENCE FOR VALIDATION -- read this before trying again. The only controlled
# precision pair with memory (jobs 1236/1238) is a LoRA run, and this calculator
# models FULL pretraining; there is no LoRA or PEFT mode anywhere in it. The frozen
# arm proves those regimes differ in a load-bearing way (0% vs 29.2% activation
# saving), so that pair CANNOT validate the activation-precision term at all.
#
# The weight term does transfer -- it is identical in both arms -- and it still does
# not reconcile. At the pair's f = 0.50 it predicts 2.0*(1-0.5)*8.825e9 = +8.825 GB
# against a measured delta of +3.300 GB (2.67x), now with no activation saving
# available to offset it. Matching exactly would need a(f=0) = 0.7479 B/param.
# (Superseded figures: the earlier 3.44x and 18.165 GB used a round 70.0e9 parameter
# count; the catalog says 70.6e9, giving 8.825e9/GPU and 17.650 GB of BF16 weights.)
#
# The one mechanism still untested is PIPELINED residency. Job 349's layer sweep
# refuted "layers in flight" only for a NON-pipelined forward-then-backward, where
# every layer's copy is necessarily alive at once; it says nothing about an
# interleaved 1F1B schedule, where a chunk's fp8 copy need only live from that
# chunk's forward to its backward, so the resident count is bounded by chunks in
# flight rather than by all 40 local layers. The probe has no pipeline parallelism
# and so STRUCTURALLY cannot measure this. What would settle it is a controlled
# bf16/fp8 pair on a FULL PRETRAINING recipe at PP=1 (where f is ~1.25%, so recompute
# is nearly inert and the weight term is near its f=0 value). Until then, shipping
# a = 2.0*(1-f) would overstate a 70B pretrain at TP=4/PP=1 by tens of GB on the
# strength of a constant whose only end-to-end test misses by 2.67x -- the Option C
# error with better measurements behind it.
ACT_BYTES_PER_ELEMENT = 2  # activations are resident in BF16 at either PRECISION
SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER = 0.45  # Reduction from selective checkpointing

# --- Activation recomputation -------------------------------------------------
# Megatron/NeMo recipes commonly set, via overrides.py:
#     recompute_granularity="full", recompute_method="block", recompute_num_layers=N
# which fully recomputes N layers of every pipeline chunk instead of storing their
# activations. A recomputed block stores only its own INPUT, not the ~6 intermediate
# tensors the multiplier above accounts for.
#
# This matters because the fraction recomputed depends on PP and VP, not just on N:
# layers per chunk is layers_per_gpu / VP under an interleaved schedule, so
#     PP=4 VP=20 -> 80/(4*20) = 1 layer/chunk  -> 1 of 1  -> ~100% recomputed
#     PP=2 VP=20 -> 80/(2*20) = 2 layers/chunk -> 1 of 2  -> ~50%  recomputed
#     PP=1 no VP -> 80 layers in one stage     -> 1 of 80 -> ~1.25% recomputed
# Changing PP therefore silently changes activation memory by a large factor.
#
# DEFAULT IS INERT: granularity None recomputes nothing, which is exactly the
# behaviour before this was added. Set it to model a recipe that recomputes.
RECOMPUTE_GRANULARITY = None  # None | "full" -- "full" recomputes whole blocks
RECOMPUTE_NUM_LAYERS = 0  # layers recomputed per pipeline chunk (recompute_num_layers)
RECOMPUTE_BOUNDARY_BYTES_PER_ELEMENT = 2  # a recomputed block still stores its input

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

# Training step boundaries
#
# NO LONGER USED TO CLASSIFY BATCH CONFIGURATIONS. Phase 2 used to score each
# configuration against these bands plus a priority-0 "matches TOKENS_PER_BATCH"
# check; both were unusable and were replaced by a RELATIVE rule
# (phase2_batch.classify_step_outliers): Phase 2 now reports the step count a
# configuration implies and marks only the smallest and largest step counts within
# a (variant, hardware) pair as outliers.
#
# Why the absolute bands went:
#   - The priority-0 target check was UNREACHABLE. tokens_per_batch is
#     dp*micro*accum*seq_len and cannot land within +/-MARGIN (2%) of an arbitrary
#     absolute number: 0 hits across 14,080 swept configurations, so priority 0 was
#     never returned and the optimal_configs list was always empty.
#   - The bands DISAGREED with TOKENS_PER_BATCH. TOTAL_TOKENS / TOKENS_PER_BATCH =
#     3,750,000 steps, 3.1x MAX_STEPS, while the optimal band implied a batch of
#     18.8M-50.0M tokens, 5x-12x the configured target.
#   - The verdict was useless in practice: 80.8% of configurations landed in "too
#     many steps". Switching to the relative rule moved 10,076 of those to
#     "reasonable" without anything about them changing -- which is the point, but
#     also means "reasonable" now asserts only "not extreme among its siblings".
#
# STILL LIVE: MIN_STEPS is the step ceiling in compute_max_tokens_per_batch below,
# and MIN_STEPS/MAX_STEPS bound the reported band in describe_batch_budget. The four
# LOWER_/UPPER_BOUND_* constants are now unused by Phase 2; they are left in place
# rather than deleted because removing them is a separate decision.
#
# THE ABSOLUTE AXIS DID COME BACK, but as one credibility floor rather than as
# quality bands: compute_absurd_step_ceiling flags a configuration implying more
# steps than TOTAL_TOKENS / TOKENS_PER_BATCH (3,750,000), i.e. a batch smaller than
# Llama 3's ramp-start. It is 3.1x looser than MAX_STEPS, so it does not restore
# the old "too many steps" verdict, and it is reported alongside the relative
# verdict instead of replacing it. These six band constants remain unused by it.
LOWER_BOUND_OPTIM_RANGE = 300_000
UPPER_BOUND_OPTIM_RANGE = 800_000
LOWER_BOUND_LOW_RANGE = 200_000
UPPER_BOUND_LOW_RANGE = 300_000
LOWER_BOUND_HIGH_RANGE = 800_000
UPPER_BOUND_HIGH_RANGE = 1_200_000
MIN_STEPS = 200_000
MAX_STEPS = 1_200_000

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


def compute_absurd_step_ceiling(total_tokens, ref_batch_tokens=None):
    """Step count above which a configuration is outside published practice.

    This is an ABSOLUTE sanity check, and it exists because the relative outlier
    rule in phase2_batch.classify_step_outliers structurally cannot provide one.
    That rule judges a configuration only against its siblings in the same
    (variant, hardware) pair, so a pair whose ENTIRE range is absurd reports every
    middle configuration as "reasonable". Measured: Llama-3.1-8B on "1 g5" spans
    28,610,229 to 3,662,109,375 steps, and the relative rule calls the 57,220,458
    to 1,831,054,687 middle band reasonable. Nothing in a relative comparison can
    object, because within that pair those really are the least extreme values.

    THE REFERENCE IS THE SMALLEST BATCH IN PUBLISHED PRACTICE, not a quality
    target. TOKENS_PER_BATCH (4e6) is Llama 3 405B's INITIAL batch -- the smallest
    batch it used at any point in training, before ramping to 8M and then 16M
    (arXiv:2407.21783 S3.4.1). DeepSeek-V3's smallest was larger still, 3072 x 4096
    = 12.6M (arXiv:2412.19437 S4.2). So a configuration implying MORE steps than
    total_tokens / 4e6 asks for a batch smaller than any published frontier run
    used at any point, which is a statement about the published envelope rather
    than an opinion about optimality. At TOTAL_TOKENS = 15e12 that is 3,750,000
    steps.

    This deliberately does NOT reinstate the bands that were removed. The bands
    graded quality on an absolute scale and condemned 81% of configurations as
    "too many steps" while telling the reader nothing. This returns one
    order-of-magnitude floor on credibility, is reported as a SEPARATE axis from
    the relative verdict rather than overriding it, and is 3.1x looser than
    MAX_STEPS -- so a configuration can be well outside the old "optimal" window
    and still not be absurd.

    WHAT IT CATCHES, measured over the full 13,728-configuration post-cap sweep:
    8,096 configurations (59.0%), but the distribution is what matters and it is
    sharply targeted rather than a blanket condemnation --

        g5 / g6 / g6e (inference-class)   95.0% flagged
        p6e-gb200                         75.0% flagged
        p5 / p5en / p6-b200 / p6-b300     26.3% flagged

    and of the 704 pairs, the 132 whose every configuration is absurd are EXACTLY
    the six small presets (1 and 4 instances of g5, g6, g6e) x 22 variants. No
    p5-class or p6-class pair is wholly absurd. The check is therefore mostly
    answering "this topology is too small to train this token budget at all",
    which is the case the relative rule is blind to.

    THERE IS NO LOWER COUNTERPART, and that is by construction rather than an
    omission. The caller admits a configuration only when tokens_per_batch is
    strictly below compute_max_tokens_per_batch, whose step ceiling term is
    total_tokens / MIN_STEPS. So steps = total_tokens / batch > MIN_STEPS always
    holds for any configuration Phase 2 ever sees -- zero of the 13,728 fall below
    MIN_STEPS, and none can. The too-few-steps side is already enforced upstream
    by the derived cap, so adding a second check for it would be dead code.

    Args:
        total_tokens: Training token budget (TOTAL_TOKENS).
        ref_batch_tokens: Smallest batch considered credible, in tokens. Defaults
            to TOKENS_PER_BATCH, which is Llama 3's ramp-start batch.

    Returns:
        Step count above which a configuration is flagged absurd.
    """
    if ref_batch_tokens is None:
        ref_batch_tokens = TOKENS_PER_BATCH
    if total_tokens <= 0 or ref_batch_tokens <= 0:
        raise ValueError(
            f"total_tokens and ref_batch_tokens must be positive, got "
            f"total_tokens={total_tokens}, ref_batch_tokens={ref_batch_tokens}"
        )
    return total_tokens / ref_batch_tokens


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
