"""
Phase 8: Inference sizing -- KV cache, TTFT, TPOT, concurrency and SLO solving.

This phase did not exist. Repo-wide, `serve`, `serving`, `TTFT`, `TPOT`,
`latency`, `kv_cache`, `prefill` and `decode` were all 0 hits, so the calculator
could size a training run and nothing else -- two of the three workloads it is
meant to cover were absent.

WHY INFERENCE IS DIFFERENT PHYSICS, NOT A VARIANT OF TRAINING
-------------------------------------------------------------
Serving has two regimes with DIFFERENT bottlenecks, and conflating them is the
usual way a sizing estimate goes wrong:

  PREFILL (sets TTFT) is COMPUTE-bound. The whole prompt is processed in one
  parallel forward pass, so it saturates the tensor cores and scales with
  FLOPs: 2 FLOPs per active parameter per token (forward only -- there is no
  backward pass, which is why it is 2N and not the 6N of training).

  DECODE (sets TPOT) is MEMORY-BANDWIDTH-bound. Generating one token requires
  reading EVERY weight once and the whole KV cache once, to do an amount of
  arithmetic proportional to a single token. Arithmetic intensity is therefore
  about 1 FLOP per byte read, orders of magnitude below what the tensor cores
  want, so the GPU stalls on HBM and peak TFLOPS is irrelevant. TPOT is
  (bytes that must be read) / (HBM bandwidth).

That is why this module needs `hbm_bw_gbps` on the hardware -- a field the
config did not carry, because nothing here was bandwidth-bound before -- and why
a decode estimate that divides FLOPs by peak TFLOPS is wrong by orders of
magnitude.

There are also no gradients, no optimizer state and no activation checkpointing,
so serving memory is weights + KV cache + a small workspace. A 70B model needs
~140 GB to serve in BF16 and ~1.1 TB to train.

KV CACHE, AND THE MLA TRAP
--------------------------
For standard MHA/GQA the cache is

    2 (K and V) * layers * kv_heads * head_dim * dtype_bytes   per token

Multi-head Latent Attention (the DeepSeek family) does NOT cache per-head K and
V -- it caches a single compressed latent per token, of width
kv_lora_rank + qk_rope_head_dim. Applying the GQA formula to DeepSeek-V3 reads
its kv_heads=128 and returns 1708 KiB/token against a real 68.6 KiB/token, a
24.9x over-estimate, because those head counts describe the attention COMPUTE
shape and not what is stored. This module keys on `kv_lora_rank` and refuses to
guess: a variant that looks like MLA but carries no compression fields is
returned FLAGGED as an upper bound rather than silently mis-sized.
"""

# Import configurations
from configuration.project_config import *
from configuration.advanced_config import *


# Fraction of peak HBM bandwidth a real decode kernel achieves. Decode is a
# stream of large contiguous weight reads, which is the friendliest possible
# access pattern, but it is not perfect: published vLLM/TensorRT-LLM decode
# throughput lands around 0.7-0.85 of peak. Named and applied once so it is
# visible and tunable rather than buried in an estimate.
DECODE_BW_UTILIZATION = 0.80

# Fraction of peak FLOPS a real prefill achieves. Prefill is a dense GEMM over
# the whole prompt, so it behaves like a training forward pass.
PREFILL_MFU = 0.45

# Serving workspace beyond weights and KV: CUDA context, activation buffers for
# the in-flight batch, fragmentation. Expressed as GB per GPU rather than a
# multiplier, because it does not scale with model size.
SERVING_WORKSPACE_GB = 2.0


def kv_cache_bytes_per_token(variant, kv_dtype_bytes=2):
    """KV cache bytes for ONE token of context, whole model (before TP).

    Returns (bytes_per_token, kind, flagged) where `kind` is "mla", "gqa" or
    "gqa-upper-bound" and `flagged` is True when the figure is not trustworthy.
    """
    layers = variant["layers"]

    # Explicit catalog flag wins over any formula: a variant whose attention
    # shape is unpublished gets its number FLAGGED, not quietly computed.
    unverified = bool(variant.get("kv_cache_unverified"))

    # MLA: one compressed latent per token per layer, not per-head K and V.
    if variant.get("kv_lora_rank"):
        width = variant["kv_lora_rank"] + variant.get("qk_rope_head_dim", 0)
        # The latent is a single tensor, not a K/V pair -- no factor of 2.
        return layers * width * kv_dtype_bytes, "mla", unverified

    kv_heads = variant.get("kv_heads")
    q_heads = variant.get("q_heads")
    head_dim = variant.get("head_dim")
    if not head_dim and q_heads:
        head_dim = variant["d"] // q_heads

    if not kv_heads or not head_dim:
        return 0, "unknown", True

    naive = 2 * layers * kv_heads * head_dim * kv_dtype_bytes

    # Looks like MLA (asymmetric qk/v projection widths) but carries no
    # compression fields: the GQA formula is an UPPER BOUND here, not an answer.
    if variant.get("qk_head_dim") and variant.get("v_head_dim") and (
        variant["qk_head_dim"] != variant["v_head_dim"]
    ):
        return naive, "gqa-upper-bound", True

    return naive, ("gqa-unverified" if unverified else "gqa"), unverified


def calculate_inference_memory(
    variant,
    gpu_mem_gb,
    tp=1,
    pp=1,
    weight_dtype_bytes=2,
    kv_dtype_bytes=2,
    context_len=4096,
    workspace_gb=SERVING_WORKSPACE_GB,
    util_threshold=0.90,
):
    """Serving memory per GPU, and how many concurrent sequences fit.

    No gradients and no optimizer state: that is the whole reason a model can be
    served on a fraction of the GPUs it needs to train.
    """
    total_params = variant.get("total_params_B", 0) * 1e9
    weight_bytes = total_params * weight_dtype_bytes

    # Weights shard across the full serving replica (TP within a stage, PP
    # across stages), so the per-GPU share is the model over tp*pp.
    shards = max(1, tp) * max(1, pp)
    weight_gb_per_gpu = weight_bytes / shards / 1e9

    kv_per_token, kv_kind, kv_flagged = kv_cache_bytes_per_token(
        variant, kv_dtype_bytes
    )
    # The KV cache shards with TP (each rank holds its own heads / latent slice)
    # and with PP (each stage caches only its own layers).
    kv_per_seq_gb = kv_per_token * context_len / shards / 1e9

    usable_gb = gpu_mem_gb * util_threshold
    available_for_kv = usable_gb - weight_gb_per_gpu - workspace_gb
    fits = available_for_kv > 0
    max_concurrent = int(available_for_kv / kv_per_seq_gb) if (
        fits and kv_per_seq_gb > 0
    ) else 0

    return {
        "weight_gb_per_gpu": weight_gb_per_gpu,
        "kv_bytes_per_token": kv_per_token,
        "kv_kind": kv_kind,
        "kv_flagged": kv_flagged,
        "kv_gb_per_sequence": kv_per_seq_gb,
        "workspace_gb": workspace_gb,
        "usable_gb": usable_gb,
        "available_for_kv_gb": available_for_kv,
        "weights_fit": fits,
        "max_concurrent_sequences": max_concurrent,
        "shards": shards,
    }


def calculate_ttft_ms(
    variant,
    prompt_len,
    peak_tflops_per_gpu,
    gpus_in_replica,
    prefill_mfu=PREFILL_MFU,
    attention_flops_factor=ATTENTION_FLOPS_FACTOR,
):
    """Time to first token, ms. COMPUTE-bound: prefill is a dense GEMM.

    2 FLOPs per active parameter per prompt token -- forward only, no backward,
    which is why this is 2N and training is 6N. The O(prompt_len^2) attention
    term is added at half the training factor for the same reason.
    """
    active = variant.get("active_params_B", variant.get("total_params_B", 0)) * 1e9
    linear_flops = 2.0 * active * prompt_len

    attn_flops = 0.0
    if attention_flops_factor and variant.get("layers") and variant.get("d"):
        # Training charges `factor` for fwd+bwd; prefill is forward only.
        attn_flops = (
            (attention_flops_factor / 3.0)
            * variant["layers"] * prompt_len * prompt_len * variant["d"]
        )

    achieved = peak_tflops_per_gpu * gpus_in_replica * prefill_mfu * 1e12
    if achieved <= 0:
        return float("inf")
    return (linear_flops + attn_flops) / achieved * 1000.0


def calculate_tpot_ms(
    variant,
    hbm_bw_gbps,
    gpus_in_replica,
    context_len=4096,
    batch_size=1,
    weight_dtype_bytes=2,
    kv_dtype_bytes=2,
    bw_utilization=DECODE_BW_UTILIZATION,
):
    """Time per output token, ms. MEMORY-BANDWIDTH-bound, not compute-bound.

    One decode step reads every weight once plus the KV cache of every sequence
    in the batch, to do roughly one token's arithmetic. So the step time is
    bytes-read over bandwidth, and peak TFLOPS does not enter.

    Weights are read once per step regardless of batch size, which is exactly
    why batching raises throughput without raising per-token latency much -- the
    weight read amortises across the batch.
    """
    total_params = variant.get("total_params_B", 0) * 1e9
    # MoE: only the ACTIVE experts are read on a given token.
    active_params = variant.get("active_params_B", total_params / 1e9) * 1e9
    weight_bytes = active_params * weight_dtype_bytes

    kv_per_token, _, _ = kv_cache_bytes_per_token(variant, kv_dtype_bytes)
    kv_bytes = kv_per_token * context_len * batch_size

    bytes_read = weight_bytes + kv_bytes
    aggregate_bw = hbm_bw_gbps * gpus_in_replica * bw_utilization * 1e9
    if aggregate_bw <= 0:
        return float("inf")
    return bytes_read / aggregate_bw * 1000.0


def size_for_slo(
    variant,
    hw,
    target_ttft_ms=None,
    target_tpot_ms=None,
    prompt_len=4096,
    output_len=512,
    required_concurrency=1,
    tp_candidates=(1, 2, 4, 8),
    precision="BF16",
):
    """Smallest GPU count in one replica that meets the SLOs. THE solver.

    This is what the calculator was missing: the other phases evaluate a
    configuration you supply, where sizing means INVERTING an SLO into a
    topology. Returns every feasible (tp, gpus) with its metrics, smallest
    first, plus the binding constraint when nothing fits.
    """
    peak = hw.get(
        "peak_tflops_fp8" if precision == "FP8" else "peak_tflops_bf16", 0
    )
    hbm = hw.get("hbm_bw_gbps", 0)
    gpu_mem = hw.get("mem_gb", 0)
    context_len = prompt_len + output_len

    feasible, rejected = [], []
    for tp in sorted(tp_candidates):
        mem = calculate_inference_memory(
            variant, gpu_mem, tp=tp, pp=1, context_len=context_len,
        )
        if not mem["weights_fit"]:
            rejected.append((tp, "weights do not fit"))
            continue
        if mem["max_concurrent_sequences"] < required_concurrency:
            rejected.append((
                tp,
                f"fits only {mem['max_concurrent_sequences']} concurrent, "
                f"need {required_concurrency}",
            ))
            continue

        ttft = calculate_ttft_ms(variant, prompt_len, peak, tp)
        tpot = calculate_tpot_ms(
            variant, hbm, tp, context_len=context_len,
            batch_size=required_concurrency,
        )
        if target_ttft_ms is not None and ttft > target_ttft_ms:
            rejected.append((tp, f"TTFT {ttft:.0f}ms > target {target_ttft_ms:.0f}ms"))
            continue
        if target_tpot_ms is not None and tpot > target_tpot_ms:
            rejected.append((tp, f"TPOT {tpot:.1f}ms > target {target_tpot_ms:.1f}ms"))
            continue

        feasible.append({
            "tp": tp,
            "gpus_per_replica": tp,
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "tokens_per_second": (
                1000.0 / tpot * required_concurrency if tpot > 0 else 0.0
            ),
            "max_concurrent_sequences": mem["max_concurrent_sequences"],
            "weight_gb_per_gpu": mem["weight_gb_per_gpu"],
            "kv_gb_per_sequence": mem["kv_gb_per_sequence"],
            "kv_kind": mem["kv_kind"],
            "kv_flagged": mem["kv_flagged"],
        })

    return {
        "variant": variant["name"],
        "hardware": hw.get("name"),
        "feasible": feasible,
        "recommended": feasible[0] if feasible else None,
        "rejected": rejected,
    }
