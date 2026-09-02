"""
Phase 9: Decode (Token Generation) Analysis for LLM Inference.

Calculates the decode phase performance, which determines ITL (Inter-Token
Latency) and aggregate throughput. The decode phase is primarily
memory-bandwidth bound — each token generation requires reading all model
weights plus the KV cache from HBM.

Key formulas:
  model_bytes = total_params_B * 1e9 * quant_bytes / TP
  kv_bytes = kv_per_token_bytes * seq_len  (grows as tokens are generated)
  bytes_per_step = model_bytes + kv_bytes
  max_tok_s_single = hbm_bw_gbps * 1e9 / bytes_per_step * bw_efficiency
  itl_ms = 1000 / max_tok_s_single

Batched decode:
  bytes_batched = model_bytes + batch_size * kv_bytes_per_request
  tokens_per_read = batch_size  (one token per request per step)
  throughput_tok_s = tokens_per_read / (bytes_batched / (hbm_bw * bw_eff))

Compute-bound check (high batch):
  compute_tok_s = peak_tflops * 1e12 / (2 * active_params_B * 1e9)
  actual_tok_s = min(bw_bound_tok_s, compute_bound_tok_s)

Notes:
  - Uses active_params_B for compute-bound check (MoE: only active experts).
  - Uses total_params_B for model weight bytes (all weights loaded in HBM).
  - Imports calculate_kv_per_token_bytes from phase7_kv_cache.
  - bw_efficiency from engine_mfu["decode_bw_eff"] (typically 0.70-0.80).
  - batching_eff from engine_mfu["batching_eff"] (typically 0.85-0.90).
  - At high batch sizes decode transitions from memory-bound to compute-bound.
"""

import math
import csv

from utils.phase7_kv_cache import calculate_kv_per_token_bytes


def _select_peak_tflops(hw, quant_bytes):
    """Select the appropriate peak TFLOPS based on quantization.

    If quant_bytes <= 1 (FP8), use fp8_tflops from hardware config.
    Falls back to peak_tflops_bf16 when fp8_tflops is 0 (e.g., A10G).
    For quant_bytes == 2 (BF16), always uses peak_tflops_bf16.

    Args:
        hw: Hardware config dict (has fp8_tflops, peak_tflops_bf16).
        quant_bytes: Bytes per parameter (1 for FP8, 2 for BF16).

    Returns:
        Peak TFLOPS per GPU as a float.
    """
    if quant_bytes <= 1:
        fp8 = hw.get("fp8_tflops", 0)
        if fp8 > 0:
            return float(fp8)
    return float(hw["peak_tflops_bf16"])


def calculate_decode_bytes_per_step(variant, hw, tp, quant_bytes, kv_dtype_bytes, seq_len):
    """Bytes read from HBM for one decode step (one token) at given seq_len.

    During decode, each token generation requires:
      1. Reading all model weights from HBM (sharded by TP).
      2. Reading the KV cache for the current sequence length.

    Formulas:
        model_bytes = total_params_B * 1e9 * quant_bytes / TP
        kv_bytes = kv_per_token_bytes * seq_len
        total_bytes = model_bytes + kv_bytes

    Args:
        variant: Model variant dict.
        hw: Hardware config dict (unused, reserved for future extensions).
        tp: Tensor parallelism degree.
        quant_bytes: Bytes per model parameter (1 for FP8, 2 for BF16).
        kv_dtype_bytes: Bytes per KV cache element (1 for FP8, 2 for BF16).
        seq_len: Current sequence length (prompt + generated tokens so far).

    Returns:
        Dict with:
        - model_bytes: Model weight bytes per GPU.
        - kv_bytes: KV cache bytes per GPU for this sequence length.
        - total_bytes: Total bytes read from HBM per decode step.
    """
    # Model weights: all parameters sharded by TP
    total_params = variant["total_params_B"] * 1e9
    model_bytes = (total_params * quant_bytes) / tp

    # KV cache: per-token bytes (across all layers) * current sequence length
    kv_per_token = calculate_kv_per_token_bytes(variant, tp, kv_dtype_bytes)
    kv_bytes = kv_per_token * seq_len

    total_bytes = model_bytes + kv_bytes

    return {
        "model_bytes": model_bytes,
        "kv_bytes": kv_bytes,
        "total_bytes": total_bytes,
    }


def calculate_single_request_throughput(variant, hw, tp, quant_bytes, kv_dtype_bytes,
                                        isl, osl, bw_efficiency,
                                        tp_comm_ms_per_step=0.0):
    """Max tokens/sec for a single request (batch=1).

    Averages over the decode phase where seq_len grows from ISL to ISL+OSL.
    Uses midpoint approximation: avg_seq_len = ISL + OSL / 2.

    The throughput is memory-bandwidth bound:
        bytes_per_step = model_bytes + kv_per_token * avg_seq_len
        tok_s = hbm_bw_gbps * 1e9 * bw_efficiency / bytes_per_step

    When TP > 1, adds per-step overhead from:
        - TP communication (all-reduce latency)
        - Per-layer kernel launch overhead (CUDA kernel dispatch)
        - Synchronization stalls between compute and communication

    Args:
        variant: Model variant dict.
        hw: Hardware config dict (has hbm_bw_gbps).
        tp: Tensor parallelism degree.
        quant_bytes: Bytes per parameter (1 for FP8, 2 for BF16).
        kv_dtype_bytes: Bytes per KV element.
        isl: Input sequence length (prompt tokens).
        osl: Output sequence length (tokens to generate).
        bw_efficiency: HBM bandwidth efficiency (0.0-1.0).
        tp_comm_ms_per_step: TP communication overhead per decode step (ms).

    Returns:
        Dict with:
        - avg_tok_s: Average tokens/sec over the decode phase.
        - itl_ms: Average inter-token latency in milliseconds.
        - avg_seq_len: Midpoint sequence length used for estimate.
        - bytes_per_step: Average bytes read per decode step.
        - bottleneck: "memory" or "tp_comm".
    """
    # Midpoint approximation: seq_len grows from ISL to ISL+OSL during decode
    avg_seq_len = isl + osl / 2.0

    # Bytes per decode step at the average sequence length
    step_info = calculate_decode_bytes_per_step(
        variant, hw, tp, quant_bytes, kv_dtype_bytes, avg_seq_len
    )
    bytes_per_step = step_info["total_bytes"]

    # Memory-bandwidth-bound throughput: one token per step
    hbm_bw_gbps = hw.get("hbm_bw_gbps", 3350)
    effective_bw = hbm_bw_gbps * 1e9 * bw_efficiency

    if bytes_per_step > 0 and effective_bw > 0:
        mem_step_time_s = bytes_per_step / effective_bw

        # Per-layer kernel launch overhead: each transformer layer requires
        # ~4-6 CUDA kernels (QKV proj, attention, output proj, gate+up, down).
        # With TP=1 kernels pipeline well; with TP>1 synchronization barriers
        # between compute and NCCL kernels reduce pipelining effectiveness.
        layers = variant.get("layers", 48)
        kernels_per_layer = 5
        if tp <= 1:
            # TP=1: CUDA streams pipeline kernels efficiently, minimal overhead
            kernel_launch_us = 2.0
        else:
            # TP>1: synchronization barriers between compute & NCCL stall pipeline
            # Plus NCCL kernels themselves have higher dispatch overhead
            kernel_launch_us = 10.0
        kernel_overhead_s = layers * kernels_per_layer * kernel_launch_us / 1e6

        # Add TP communication overhead (serialized per decode step)
        total_step_time_s = mem_step_time_s + (tp_comm_ms_per_step / 1000.0) + kernel_overhead_s
        avg_tok_s = 1.0 / total_step_time_s if total_step_time_s > 0 else 0.0
        bottleneck = "tp_comm" if tp_comm_ms_per_step / 1000.0 > mem_step_time_s else "memory"
    else:
        avg_tok_s = 0.0
        bottleneck = "memory"

    itl_ms = (1000.0 / avg_tok_s) if avg_tok_s > 0 else float("inf")

    return {
        "avg_tok_s": round(avg_tok_s, 2),
        "itl_ms": round(itl_ms, 3),
        "avg_seq_len": avg_seq_len,
        "bytes_per_step": bytes_per_step,
        "bottleneck": bottleneck,
    }


def _compute_bound_throughput(variant, hw, tp, quant_bytes, batch_size):
    """Compute-bound throughput ceiling for batched decode.

    At high batch sizes, the decode phase can become compute-bound
    rather than memory-bound. This calculates the compute ceiling.

    Formula:
        flops_per_token = 2 * active_params_B * 1e9
        peak_tflops = select_peak(hw, quant_bytes)
        compute_tok_s_per_gpu = peak_tflops * 1e12 / flops_per_token
        compute_tok_s_total = compute_tok_s_per_gpu * TP

    Note: batch_size is used at the call site to compare against
    memory-bandwidth throughput. The compute bound is the same
    regardless of batch_size — it's the GPU's raw compute capacity.

    Args:
        variant: Model variant dict (has active_params_B).
        hw: Hardware config dict.
        tp: Tensor parallelism degree.
        quant_bytes: Bytes per parameter.
        batch_size: Current batch size (for context; compute bound is fixed).

    Returns:
        Compute-bound tokens/sec (total across TP GPUs).
    """
    active_params = variant["active_params_B"] * 1e9
    flops_per_token = 2.0 * active_params
    peak_tflops = _select_peak_tflops(hw, quant_bytes)

    if flops_per_token > 0 and peak_tflops > 0:
        # Compute capacity per GPU in tokens/sec
        compute_tok_s_per_gpu = (peak_tflops * 1e12) / flops_per_token
        # TP GPUs work in parallel on different weight shards
        return compute_tok_s_per_gpu * tp
    return 0.0


def calculate_batched_throughput(variant, hw, tp, quant_bytes, kv_dtype_bytes,
                                 isl, osl, batch_size, bw_efficiency, batching_eff,
                                 tp_comm_ms_per_step=0.0):
    """Aggregate tokens/sec with batch_size concurrent requests.

    With batching, model weights are read ONCE from HBM and amortized
    across all batch_size requests. Each request contributes its own
    KV cache reads. One token is generated per request per step.

    Memory-bandwidth bound:
        bytes_batched = model_bytes + batch_size * kv_bytes_per_request
        tokens_per_read = batch_size
        bw_tok_s = tokens_per_read * hbm_bw * bw_eff / bytes_batched

    Compute bound (checked at high batch):
        compute_tok_s = peak_tflops * 1e12 / (2 * active_params_B * 1e9)

    The effective throughput is min(bw_tok_s, compute_tok_s), scaled
    by batching_eff to account for continuous batching overhead.

    Uses midpoint approximation: avg_seq_len = ISL + OSL / 2.

    Args:
        variant: Model variant dict.
        hw: Hardware config dict.
        tp: Tensor parallelism degree.
        quant_bytes: Bytes per parameter.
        kv_dtype_bytes: Bytes per KV element.
        isl: Input sequence length.
        osl: Output sequence length.
        batch_size: Number of concurrent requests.
        bw_efficiency: HBM bandwidth efficiency.
        batching_eff: Continuous batching efficiency (0.0-1.0).

    Returns:
        Dict with:
        - total_tok_s: Aggregate tokens/sec across all requests.
        - per_request_tok_s: Tokens/sec per individual request.
        - itl_ms: Inter-token latency per request (ms).
        - bw_bound_tok_s: Memory-bandwidth-bound throughput.
        - compute_bound_tok_s: Compute-bound throughput ceiling.
        - bottleneck: "memory" or "compute".
        - batch_size: Effective batch size used.
        - bytes_per_step: Total bytes read per decode step.
    """
    if batch_size < 1:
        batch_size = 1

    # Midpoint approximation for KV cache size during generation
    avg_seq_len = isl + osl / 2.0

    # Model weights: read once, shared across all requests in the batch
    total_params = variant["total_params_B"] * 1e9
    model_bytes = (total_params * quant_bytes) / tp

    # KV cache: each request has its own KV cache
    kv_per_token = calculate_kv_per_token_bytes(variant, tp, kv_dtype_bytes)
    kv_bytes_per_request = kv_per_token * avg_seq_len
    kv_bytes_total = batch_size * kv_bytes_per_request

    # Total bytes read per decode step (all requests)
    bytes_per_step = model_bytes + kv_bytes_total

    # Tokens produced per step: one per request
    tokens_per_step = batch_size

    # --- Memory-bandwidth bound ---
    hbm_bw_gbps = hw.get("hbm_bw_gbps", 3350)
    effective_bw = hbm_bw_gbps * 1e9 * bw_efficiency

    if bytes_per_step > 0 and effective_bw > 0:
        mem_step_time_s = bytes_per_step / effective_bw

        # Per-layer kernel launch overhead
        layers = variant.get("layers", 48)
        kernels_per_layer = 5
        if tp <= 1:
            kernel_launch_us = 2.0
        else:
            kernel_launch_us = 10.0
        kernel_overhead_s = layers * kernels_per_layer * kernel_launch_us / 1e6

        # Add TP communication overhead (serialized per decode step)
        total_step_time_s = mem_step_time_s + (tp_comm_ms_per_step / 1000.0) + kernel_overhead_s
        bw_bound_tok_s = tokens_per_step / total_step_time_s if total_step_time_s > 0 else 0.0
    else:
        bw_bound_tok_s = 0.0

    # --- Compute bound ---
    compute_bound_tok_s = _compute_bound_throughput(
        variant, hw, tp, quant_bytes, batch_size
    )

    # The effective throughput is the minimum of both bounds,
    # scaled by batching efficiency to account for scheduler overhead
    if bw_bound_tok_s > 0 and compute_bound_tok_s > 0:
        raw_tok_s = min(bw_bound_tok_s, compute_bound_tok_s)
        bottleneck = "memory" if bw_bound_tok_s <= compute_bound_tok_s else "compute"
    elif bw_bound_tok_s > 0:
        raw_tok_s = bw_bound_tok_s
        bottleneck = "memory"
    elif compute_bound_tok_s > 0:
        raw_tok_s = compute_bound_tok_s
        bottleneck = "compute"
    else:
        raw_tok_s = 0.0
        bottleneck = "unknown"

    total_tok_s = raw_tok_s * batching_eff
    per_request_tok_s = total_tok_s / batch_size if batch_size > 0 else 0.0
    itl_ms = (1000.0 / per_request_tok_s) if per_request_tok_s > 0 else float("inf")

    return {
        "total_tok_s": round(total_tok_s, 2),
        "per_request_tok_s": round(per_request_tok_s, 2),
        "itl_ms": round(itl_ms, 3),
        "bw_bound_tok_s": round(bw_bound_tok_s, 2),
        "compute_bound_tok_s": round(compute_bound_tok_s, 2),
        "bottleneck": bottleneck,
        "batch_size": batch_size,
        "bytes_per_step": bytes_per_step,
    }


def analyze_decode(variant, hw, tp, quant_bytes, kv_dtype_bytes,
                   input_seq_lens, osl, concurrency_levels, engine_mfu,
                   max_concurrent_fn, tp_comm_ms_per_step=0.0):
    """Full decode analysis sweeping ISL and concurrency.

    For each ISL and concurrency level, calculates decode throughput and
    ITL. The effective batch size is capped by max_concurrent_fn(isl, osl)
    which returns how many requests fit in GPU memory (from Phase 7).

    Args:
        variant: Model variant dict.
        hw: Hardware config dict.
        tp: Tensor parallelism degree.
        quant_bytes: Bytes per parameter.
        kv_dtype_bytes: Bytes per KV element.
        input_seq_lens: List of ISL values to sweep.
        osl: Output sequence length.
        concurrency_levels: List of concurrency levels to test.
        engine_mfu: Engine MFU dict (decode_bw_eff, batching_eff, etc.).
        max_concurrent_fn: Callable(isl, osl) -> int, returns max concurrent
                           requests from Phase 7 memory analysis.
        tp_comm_ms_per_step: TP communication overhead per decode step in ms.
                            Calculated from Phase 10 (0.0 when TP=1).

    Returns:
        Dict with:
        - variant_name: Model name.
        - hw_name: Hardware platform name.
        - tp: Tensor parallelism degree.
        - quant_bytes: Quantization bytes.
        - kv_dtype_bytes: KV cache dtype bytes.
        - osl: Output sequence length.
        - bw_efficiency: Decode bandwidth efficiency used.
        - batching_eff: Batching efficiency used.
        - tp_comm_ms_per_step: TP overhead used.
        - isl_results: List of per-ISL dicts, each containing:
          - isl: Input sequence length.
          - max_concurrent: Max concurrent from Phase 7.
          - single_request: Single-request throughput dict.
          - concurrency_results: List of per-concurrency dicts.
    """
    bw_efficiency = engine_mfu.get("decode_bw_eff", 0.70)
    batching_eff = engine_mfu.get("batching_eff", 0.85)

    isl_results = []
    for isl in input_seq_lens:
        # Max concurrent requests that fit in GPU memory (from Phase 7)
        max_concurrent = max_concurrent_fn(isl, osl)

        # Single-request baseline (batch=1)
        single_req = calculate_single_request_throughput(
            variant, hw, tp, quant_bytes, kv_dtype_bytes,
            isl, osl, bw_efficiency, tp_comm_ms_per_step
        )

        # Sweep concurrency levels
        concurrency_results = []
        for conc in concurrency_levels:
            # Effective batch is capped by memory capacity
            effective_batch = min(conc, max_concurrent) if max_concurrent > 0 else 0

            if effective_batch < 1:
                concurrency_results.append({
                    "requested_concurrency": conc,
                    "effective_batch": 0,
                    "total_tok_s": 0.0,
                    "per_request_tok_s": 0.0,
                    "itl_ms": float("inf"),
                    "bw_bound_tok_s": 0.0,
                    "compute_bound_tok_s": 0.0,
                    "bottleneck": "oom",
                    "memory_limited": True,
                })
                continue

            batched = calculate_batched_throughput(
                variant, hw, tp, quant_bytes, kv_dtype_bytes,
                isl, osl, effective_batch, bw_efficiency, batching_eff,
                tp_comm_ms_per_step
            )

            concurrency_results.append({
                "requested_concurrency": conc,
                "effective_batch": effective_batch,
                "total_tok_s": batched["total_tok_s"],
                "per_request_tok_s": batched["per_request_tok_s"],
                "itl_ms": batched["itl_ms"],
                "bw_bound_tok_s": batched["bw_bound_tok_s"],
                "compute_bound_tok_s": batched["compute_bound_tok_s"],
                "bottleneck": batched["bottleneck"],
                "memory_limited": effective_batch < conc,
            })

        isl_results.append({
            "isl": isl,
            "max_concurrent": max_concurrent,
            "single_request": single_req,
            "concurrency_results": concurrency_results,
        })

    return {
        "variant_name": variant["name"],
        "hw_name": hw["name"],
        "tp": tp,
        "quant_bytes": quant_bytes,
        "kv_dtype_bytes": kv_dtype_bytes,
        "osl": osl,
        "bw_efficiency": bw_efficiency,
        "batching_eff": batching_eff,
        "tp_comm_ms_per_step": tp_comm_ms_per_step,
        "active_params_B": variant["active_params_B"],
        "total_params_B": variant["total_params_B"],
        "isl_results": isl_results,
    }


def generate_decode_recommendations(all_results):
    """Generate recommendations based on decode analysis.

    Examines ITL and throughput values across all hardware/variant/ISL
    combinations and produces actionable guidance.

    Args:
        all_results: Dict mapping hw_name -> list of variant decode analyses
                     (each from analyze_decode).

    Returns:
        List of recommendation strings.
    """
    recs = []

    for hw_name, variant_results in all_results.items():
        for vr in variant_results:
            variant_name = vr["variant_name"]

            for isl_r in vr["isl_results"]:
                isl = isl_r["isl"]
                single = isl_r["single_request"]
                max_conc = isl_r["max_concurrent"]

                # Single-request ITL check
                if single["itl_ms"] > 200:
                    recs.append(
                        f"High ITL: {variant_name} on {hw_name} at ISL={isl} "
                        f"has single-request ITL={single['itl_ms']:.0f} ms "
                        f"({single['avg_tok_s']:.1f} tok/s). "
                        f"Consider: higher HBM bandwidth GPU, FP8 quantization, "
                        f"or increase TP to shard model weights."
                    )
                elif single["itl_ms"] < 30:
                    recs.append(
                        f"Good: {variant_name} on {hw_name} at ISL={isl} "
                        f"achieves ITL={single['itl_ms']:.1f} ms — excellent "
                        f"for realtime conversational serving."
                    )

                # Memory-limited concurrency
                if max_conc < 1:
                    recs.append(
                        f"OOM: {variant_name} on {hw_name} at ISL={isl} "
                        f"cannot serve even a single request. Model weights "
                        f"exceed available GPU memory. Increase TP or use "
                        f"smaller quantization."
                    )
                    continue

                # Check highest concurrency result
                for cr in isl_r["concurrency_results"]:
                    if cr["memory_limited"] and cr["effective_batch"] > 0:
                        recs.append(
                            f"Memory-limited: {variant_name} on {hw_name} at "
                            f"ISL={isl}, requested concurrency={cr['requested_concurrency']} "
                            f"but only {cr['effective_batch']} fit in memory "
                            f"(max_concurrent={max_conc}). Consider: FP8 KV cache, "
                            f"larger GPU memory, or shorter sequences."
                        )

                    if cr["bottleneck"] == "compute" and cr["effective_batch"] > 0:
                        recs.append(
                            f"Compute-bound: {variant_name} on {hw_name} at "
                            f"ISL={isl}, batch={cr['effective_batch']} is "
                            f"compute-bound ({cr['total_tok_s']:.0f} tok/s). "
                            f"Higher HBM bandwidth won't help — need more "
                            f"compute (FP8, higher TP, or faster GPU)."
                        )

                    # High-throughput sweet spot
                    if (cr["total_tok_s"] > 1000
                            and cr["itl_ms"] < 100
                            and not cr["memory_limited"]):
                        recs.append(
                            f"Sweet spot: {variant_name} on {hw_name} at "
                            f"ISL={isl}, batch={cr['effective_batch']} achieves "
                            f"{cr['total_tok_s']:.0f} tok/s total with "
                            f"ITL={cr['itl_ms']:.1f} ms — good throughput/latency "
                            f"balance."
                        )

    return recs


def export_decode_results_csv(all_results, filename):
    """Export decode analysis results to CSV.

    Writes one row per (hardware, variant, ISL, concurrency) combination
    with all decode metrics: throughput, ITL, bottleneck, etc.

    Args:
        all_results: Dict mapping hw_name -> list of variant decode analyses.
        filename: Output CSV file path.
    """
    rows = []
    for hw_name, variant_results in all_results.items():
        for vr in variant_results:
            for isl_r in vr["isl_results"]:
                # Single-request row
                single = isl_r["single_request"]
                rows.append({
                    "hardware": hw_name,
                    "variant": vr["variant_name"],
                    "tp": vr["tp"],
                    "quant_bytes": vr["quant_bytes"],
                    "kv_dtype_bytes": vr["kv_dtype_bytes"],
                    "isl": isl_r["isl"],
                    "osl": vr["osl"],
                    "max_concurrent": isl_r["max_concurrent"],
                    "requested_concurrency": 1,
                    "effective_batch": 1,
                    "total_tok_s": single["avg_tok_s"],
                    "per_request_tok_s": single["avg_tok_s"],
                    "itl_ms": single["itl_ms"],
                    "bw_bound_tok_s": single["avg_tok_s"],
                    "compute_bound_tok_s": "",
                    "bottleneck": single["bottleneck"],
                    "memory_limited": False,
                })

                # Batched concurrency rows
                for cr in isl_r["concurrency_results"]:
                    rows.append({
                        "hardware": hw_name,
                        "variant": vr["variant_name"],
                        "tp": vr["tp"],
                        "quant_bytes": vr["quant_bytes"],
                        "kv_dtype_bytes": vr["kv_dtype_bytes"],
                        "isl": isl_r["isl"],
                        "osl": vr["osl"],
                        "max_concurrent": isl_r["max_concurrent"],
                        "requested_concurrency": cr["requested_concurrency"],
                        "effective_batch": cr["effective_batch"],
                        "total_tok_s": cr["total_tok_s"],
                        "per_request_tok_s": cr["per_request_tok_s"],
                        "itl_ms": cr["itl_ms"],
                        "bw_bound_tok_s": cr["bw_bound_tok_s"],
                        "compute_bound_tok_s": cr["compute_bound_tok_s"],
                        "bottleneck": cr["bottleneck"],
                        "memory_limited": cr["memory_limited"],
                    })

    if rows:
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
