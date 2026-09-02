"""
Phase 7: KV Cache Sizing and Model Memory Analysis for Inference.

Calculates:
- Model weight memory per GPU (considering TP and quantization)
- KV cache memory per token per layer
- KV cache memory per request (ISL + OSL)
- Maximum concurrent requests before OOM
- Memory breakdown: model + KV cache + engine overhead

Key formulas:
  model_mem_per_gpu = (total_params * bytes_per_param) / TP
  kv_per_token_per_layer = 2 * kv_heads_per_gpu * head_dim * kv_bytes
  kv_per_request = kv_per_token_per_layer * layers * (ISL + OSL)
  max_concurrent = floor((gpu_mem * util - model_mem - overhead) / kv_per_request)
"""

import math


def get_head_dim(variant):
    """Get the per-head dimension for a model variant.

    Uses explicit head_dim if present, otherwise derives from d / q_heads.
    Common values: 128 (most models), 256 (some DeepSeek variants).
    """
    if "head_dim" in variant and variant["head_dim"]:
        return variant["head_dim"]
    return variant["d"] // variant["q_heads"]


def calculate_model_memory_gb(variant, tp, quant_bytes):
    """Calculate model weight memory per GPU in GB.

    For dense models: total_params * bytes / TP
    For MoE models: (shared_params + expert_params_per_gpu) * bytes / TP
      where expert_params_per_gpu depends on EP (expert parallelism).
      For inference EP=1 (all experts on every GPU), TP shards everything.

    Args:
        variant: Model variant dict from variants_config.
        tp: Tensor parallelism degree.
        quant_bytes: Bytes per parameter (1 for FP8, 2 for BF16, 0.5 for INT4).

    Returns:
        Model weight memory per GPU in GB.
    """
    total_params_b = variant["total_params_B"]
    total_bytes = total_params_b * 1e9 * quant_bytes
    return total_bytes / tp / (1024**3)


def calculate_kv_per_token_bytes(variant, tp, kv_dtype_bytes):
    """Calculate KV cache memory per token across ALL layers in bytes.

    Each layer stores a K tensor and a V tensor for each KV head.
    With GQA (grouped-query attention), kv_heads < q_heads.
    With TP, KV heads are sharded across TP GPUs.

    Formula:
        kv_per_token = 2 * (kv_heads / TP) * head_dim * kv_bytes * layers

    The factor of 2 accounts for both K and V tensors.

    Args:
        variant: Model variant dict.
        tp: Tensor parallelism degree.
        kv_dtype_bytes: Bytes per KV cache element (1 for FP8, 2 for FP16/BF16).

    Returns:
        KV cache bytes per token (across all layers).
    """
    head_dim = get_head_dim(variant)
    kv_heads = variant["kv_heads"]
    layers = variant["layers"]

    # KV heads per GPU after TP sharding
    # If kv_heads < TP, each GPU still gets at least 1 KV head (replicated)
    kv_heads_per_gpu = max(1, kv_heads // tp)

    # Per token, per layer: 2 (K+V) * kv_heads_per_gpu * head_dim * bytes
    per_token_per_layer = 2 * kv_heads_per_gpu * head_dim * kv_dtype_bytes

    # Across all layers
    return per_token_per_layer * layers


def calculate_kv_per_request_gb(variant, tp, kv_dtype_bytes, isl, osl):
    """Calculate total KV cache memory per request in GB.

    A request with ISL input tokens and OSL output tokens needs KV cache
    for (ISL + OSL) tokens total — the prefill tokens plus all generated tokens.

    Args:
        variant: Model variant dict.
        tp: Tensor parallelism degree.
        kv_dtype_bytes: Bytes per KV element.
        isl: Input sequence length (prompt tokens).
        osl: Output sequence length (generated tokens).

    Returns:
        KV cache memory per request in GB.
    """
    kv_per_token = calculate_kv_per_token_bytes(variant, tp, kv_dtype_bytes)
    total_tokens = isl + osl
    return (kv_per_token * total_tokens) / (1024**3)


def calculate_max_concurrent(variant, hw, tp, quant_bytes, kv_dtype_bytes,
                             isl, osl, gpu_mem_util, engine_overhead_gb):
    """Calculate maximum concurrent requests before GPU OOM.

    Available KV memory = (GPU_mem * utilization) - model_weights - engine_overhead
    Max concurrent = floor(available / kv_per_request)

    Engine overhead includes: CUDA context, NCCL buffers, engine workspace,
    PagedAttention block tables (vLLM), radix tree (SGLang).

    Args:
        variant: Model variant dict.
        hw: Hardware config dict.
        tp: Tensor parallelism degree.
        quant_bytes: Bytes per model parameter.
        kv_dtype_bytes: Bytes per KV cache element.
        isl: Input sequence length.
        osl: Output sequence length.
        gpu_mem_util: Fraction of GPU memory available (e.g., 0.90).
        engine_overhead_gb: Fixed engine overhead in GB.

    Returns:
        Dict with memory breakdown and max concurrent requests.
    """
    gpu_mem = hw["mem_gb"]
    model_mem = calculate_model_memory_gb(variant, tp, quant_bytes)
    kv_per_req = calculate_kv_per_request_gb(variant, tp, kv_dtype_bytes, isl, osl)

    usable_mem = gpu_mem * gpu_mem_util
    available_for_kv = usable_mem - model_mem - engine_overhead_gb

    if available_for_kv <= 0:
        return {
            "fits": False,
            "gpu_mem_gb": gpu_mem,
            "usable_mem_gb": usable_mem,
            "model_mem_gb": model_mem,
            "engine_overhead_gb": engine_overhead_gb,
            "available_for_kv_gb": 0,
            "kv_per_request_gb": kv_per_req,
            "max_concurrent": 0,
            "shortage_gb": -available_for_kv,
        }

    max_conc = math.floor(available_for_kv / kv_per_req) if kv_per_req > 0 else 0

    return {
        "fits": True,
        "gpu_mem_gb": gpu_mem,
        "usable_mem_gb": round(usable_mem, 2),
        "model_mem_gb": round(model_mem, 2),
        "engine_overhead_gb": engine_overhead_gb,
        "available_for_kv_gb": round(available_for_kv, 2),
        "kv_per_request_gb": round(kv_per_req, 4),
        "max_concurrent": max_conc,
        "kv_utilization_pct": round((kv_per_req * max_conc / available_for_kv) * 100, 1) if max_conc > 0 else 0,
    }


def analyze_kv_cache(variant, hw, tp, quant_bytes, kv_dtype_bytes,
                     input_seq_lens, osl, gpu_mem_util, engine_overhead_gb=2.0):
    """Run full KV cache analysis for a variant on given hardware.

    Sweeps across all ISL values and returns memory breakdown + max concurrent
    for each.

    Args:
        variant: Model variant dict.
        hw: Hardware config dict.
        tp: Tensor parallelism degree.
        quant_bytes: Bytes per model parameter.
        kv_dtype_bytes: Bytes per KV element.
        input_seq_lens: List of ISL values to analyze.
        osl: Output sequence length.
        gpu_mem_util: GPU memory utilization fraction.
        engine_overhead_gb: Fixed engine overhead.

    Returns:
        Dict with variant info and per-ISL analysis results.
    """
    head_dim = get_head_dim(variant)
    kv_per_token = calculate_kv_per_token_bytes(variant, tp, kv_dtype_bytes)
    model_mem = calculate_model_memory_gb(variant, tp, quant_bytes)

    isl_results = []
    for isl in input_seq_lens:
        mem_info = calculate_max_concurrent(
            variant, hw, tp, quant_bytes, kv_dtype_bytes,
            isl, osl, gpu_mem_util, engine_overhead_gb
        )
        mem_info["isl"] = isl
        mem_info["osl"] = osl
        mem_info["total_seq_len"] = isl + osl
        isl_results.append(mem_info)

    return {
        "variant_name": variant["name"],
        "hw_name": hw["name"],
        "tp": tp,
        "layers": variant["layers"],
        "kv_heads": variant["kv_heads"],
        "head_dim": head_dim,
        "model_mem_gb": round(model_mem, 2),
        "kv_per_token_bytes": kv_per_token,
        "kv_per_token_kb": round(kv_per_token / 1024, 2),
        "quant_bytes": quant_bytes,
        "kv_dtype_bytes": kv_dtype_bytes,
        "isl_results": isl_results,
    }


def generate_kv_recommendations(all_results):
    """Generate recommendations based on KV cache analysis.

    Args:
        all_results: Dict mapping hw_name -> list of variant analyses.

    Returns:
        List of recommendation strings.
    """
    recs = []

    for hw_name, variant_results in all_results.items():
        for vr in variant_results:
            variant_name = vr["variant_name"]
            for isl_r in vr["isl_results"]:
                if not isl_r["fits"]:
                    recs.append(
                        f"OOM: {variant_name} does not fit on {hw_name} with TP={vr['tp']} "
                        f"(needs {isl_r.get('shortage_gb', 0):.1f} GB more). "
                        f"Consider: increase TP, use smaller quantization, or use larger GPU."
                    )
                elif isl_r["max_concurrent"] < 8:
                    recs.append(
                        f"Low concurrency: {variant_name} on {hw_name} at ISL={isl_r['isl']} "
                        f"supports max {isl_r['max_concurrent']} concurrent requests. "
                        f"KV cache per request = {isl_r['kv_per_request_gb']:.2f} GB. "
                        f"Consider: FP8 KV cache, smaller ISL, or more GPU memory."
                    )
                elif isl_r["max_concurrent"] >= 128:
                    recs.append(
                        f"Good: {variant_name} on {hw_name} at ISL={isl_r['isl']} "
                        f"supports {isl_r['max_concurrent']} concurrent requests — "
                        f"well-suited for high-concurrency serving."
                    )

    return recs


def export_kv_results_csv(all_results, filename):
    """Export KV cache analysis to CSV."""
    import csv
    rows = []
    for hw_name, variant_results in all_results.items():
        for vr in variant_results:
            for isl_r in vr["isl_results"]:
                rows.append({
                    "hardware": hw_name,
                    "variant": vr["variant_name"],
                    "tp": vr["tp"],
                    "isl": isl_r["isl"],
                    "osl": isl_r["osl"],
                    "model_mem_gb": vr["model_mem_gb"],
                    "kv_per_request_gb": isl_r["kv_per_request_gb"],
                    "available_for_kv_gb": isl_r.get("available_for_kv_gb", 0),
                    "max_concurrent": isl_r["max_concurrent"],
                    "fits": isl_r["fits"],
                })
    if rows:
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def export_kv_results_json(all_results, filename):
    """Export KV cache analysis to JSON."""
    import json
    with open(filename, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
