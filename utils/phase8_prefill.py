"""
Phase 8: Prefill (Prompt Processing) Analysis for LLM Inference.

Calculates the prefill phase performance, which determines TTFT (Time to
First Token). The prefill phase is primarily compute-bound — unlike the
decode phase which is memory-bandwidth bound.

Key formulas:
  prefill_flops = 2 * active_params_B * 1e9 * ISL
  arithmetic_intensity = prefill_flops / model_bytes_read
  machine_balance = peak_tflops * 1e12 / (hbm_bw * 1e9)
  is_compute_bound = (arithmetic_intensity > machine_balance)
  ttft_theoretical_ms = (prefill_flops / (peak_tflops * 1e12 * TP * mfu)) * 1000
  ttft_adjusted_ms = ttft_theoretical * (1 + chunked_prefill_overhead) * (1 + tp_comm_factor)

Notes:
  - Uses active_params_B (not total_params_B) for MoE models — only the
    active experts contribute to per-token compute.
  - Selects FP8 or BF16 peak TFLOPS based on quant_bytes.
  - Falls back to peak_tflops_bf16 when fp8_tflops is 0 (e.g., A10G).
  - TP communication: each layer has 2 all-reduce ops (attn + MLP) but
    during prefill this is partially overlapped. The overlap fraction comes
    from engine_mfu["tp_comm_overlap"].
"""

import math
import csv


def calculate_prefill_flops(variant, isl):
    """Calculate total FLOPs for prefilling ISL tokens.

    For the forward pass, each token requires ~2N FLOPs where N is the
    number of active parameters. For MoE models, only the active experts
    (selected by the router) contribute to per-token compute, so we use
    active_params_B instead of total_params_B.

    Formula:
        prefill_flops = 2 * active_params_B * 1e9 * ISL

    The factor of 2 comes from the multiply-accumulate operations in
    matrix multiplications (each element requires a multiply and an add).

    Args:
        variant: Model variant dict (must contain active_params_B).
        isl: Input sequence length (number of prompt tokens).

    Returns:
        Total prefill FLOPs as a float.
    """
    active_params = variant["active_params_B"] * 1e9
    return 2.0 * active_params * isl


def calculate_arithmetic_intensity(variant, isl, quant_bytes, tp):
    """Calculate ops/byte ratio to determine compute vs memory bound.

    Arithmetic intensity measures how many FLOPs are performed per byte
    of data read from HBM. During prefill the model weights are read once
    from HBM and reused across all ISL tokens, so higher ISL increases
    the intensity (making prefill more compute-bound).

    Formulas:
        flops = 2 * active_params * ISL
        bytes_read = (total_params * quant_bytes) / TP
        ops_per_byte = flops / bytes_read

    The model bytes read uses total_params_B (all weights are loaded into
    HBM) divided by TP (weights are sharded across TP GPUs).

    Args:
        variant: Model variant dict.
        isl: Input sequence length.
        quant_bytes: Bytes per parameter (1 for FP8, 2 for BF16).
        tp: Tensor parallelism degree.

    Returns:
        Dict with:
        - flops: Total prefill FLOPs.
        - bytes_read: Model bytes read from HBM per GPU.
        - ops_per_byte: Arithmetic intensity (FLOPs / byte).
    """
    flops = calculate_prefill_flops(variant, isl)
    total_params = variant["total_params_B"] * 1e9
    bytes_read = (total_params * quant_bytes) / tp
    ops_per_byte = flops / bytes_read if bytes_read > 0 else 0.0

    return {
        "flops": flops,
        "bytes_read": bytes_read,
        "ops_per_byte": ops_per_byte,
    }


def _select_peak_tflops(hw, quant_bytes):
    """Select the appropriate peak TFLOPS based on quantization.

    If quant_bytes == 1 (FP8), use fp8_tflops from the hardware config.
    If fp8_tflops is 0 (e.g., A10G which doesn't support FP8), fall back
    to peak_tflops_bf16. For quant_bytes == 2 (BF16), always use
    peak_tflops_bf16.

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


def _calculate_tp_comm_factor(variant, hw, tp, isl, quant_bytes, engine_mfu):
    """Calculate the TP communication overhead factor for prefill.

    Each transformer layer performs 2 all-reduce operations during the
    forward pass (one after attention, one after MLP). With TP > 1 these
    add latency. The engine may partially overlap communication with
    compute; the overlap fraction comes from engine_mfu["tp_comm_overlap"].

    The overhead is calculated from actual comm time vs compute time:
        comm_bytes_per_layer = 2 * 2 * (TP-1)/TP * hidden_dim * ISL * dtype_bytes
        total_comm_bytes = comm_bytes_per_layer * layers
        comm_time = total_comm_bytes / interconnect_bw + launch_latency
        compute_time = prefill_flops / (peak_tflops * TP * 1e12)
        tp_comm_factor = (1 - overlap) * comm_time / compute_time

    For NVLink (high bandwidth), this factor is small (~2-15% per rank).
    For PCIe (low bandwidth), this factor can be substantial (~50-300%).

    Args:
        variant: Model variant dict.
        hw: Hardware config dict (needs intra_node_bw_gbps, pcie_bw_gbps).
        tp: Tensor parallelism degree.
        isl: Input sequence length (affects message size for prefill).
        quant_bytes: Bytes per parameter.
        engine_mfu: Dict with tp_comm_overlap, prefill_mfu, etc.

    Returns:
        TP communication overhead factor (0.0 when TP=1).
    """
    if tp <= 1:
        return 0.0

    overlap = engine_mfu.get("tp_comm_overlap", 0.0)
    mfu = engine_mfu.get("prefill_mfu", 0.20)

    hidden_dim = variant["d"]
    layers = variant["layers"]
    active_params = variant["active_params_B"] * 1e9

    # Activation dtype is typically BF16/FP16 (2 bytes) even with FP8 weights
    act_dtype_bytes = 2

    # Comm volume per layer: 2 all-reduces, ring all-reduce factor
    message_size = hidden_dim * isl * act_dtype_bytes
    allreduce_volume_per_op = 2.0 * (tp - 1) / tp * message_size
    comm_per_layer = 2 * allreduce_volume_per_op  # 2 all-reduces
    total_comm_bytes = comm_per_layer * layers

    # Detect interconnect bandwidth
    intra_bw = hw.get("intra_node_bw_gbps", 900)
    pcie_bw = hw.get("pcie_bw_gbps", 64)
    gpus_per_node = hw.get("gpus_per_node", 8)

    if tp > gpus_per_node:
        bw_gbps = hw.get("inter_node_bw_gb", 400)
        launch_latency_us = 75.0
    elif intra_bw > 200:
        # NVLink: high-bandwidth, near-full utilization for ring all-reduce
        bw_gbps = intra_bw
        launch_latency_us = 8.0
    else:
        # PCIe: effective all-reduce bandwidth is much lower than per-link peak
        # For ring all-reduce with N GPUs over PCIe:
        # - Theoretical: PCIe_bw * (N-1)/N for each phase
        # - In practice: dual-socket CPU topology, shared PCIe root complexes,
        #   and NCCL implementation overhead reduce this significantly
        # Empirical NCCL bus bandwidth on PCIe (GB/s effective):
        #   2 GPUs: ~0.6 × per-link  (PCIe P2P, same socket)
        #   4 GPUs: ~0.15 × per-link (cross-socket, ring)
        #   8 GPUs: ~0.05 × per-link (full ring, dual-socket)
        pcie_per_link = pcie_bw
        if tp <= 2:
            pcie_factor = 0.6
        elif tp <= 4:
            pcie_factor = 0.15
        else:
            pcie_factor = 0.05  # 8-GPU ring over PCIe
        bw_gbps = pcie_per_link * pcie_factor
        launch_latency_us = 40.0

    # Comm time = bandwidth time + launch latency
    num_ops = 2 * layers
    bw_time_s = total_comm_bytes / (bw_gbps * 1e9) if bw_gbps > 0 else float("inf")
    launch_time_s = num_ops * launch_latency_us / 1e6
    comm_time_s = bw_time_s + launch_time_s

    # Compute time
    prefill_flops = 2.0 * active_params * isl
    peak_tflops = hw.get("fp8_tflops", 0) if quant_bytes <= 1 else 0
    if peak_tflops == 0:
        peak_tflops = hw["peak_tflops_bf16"]
    compute_time_s = prefill_flops / (peak_tflops * 1e12 * tp * mfu) if peak_tflops > 0 else float("inf")

    # Overhead factor
    if compute_time_s > 0 and compute_time_s != float("inf"):
        tp_comm_factor = (1.0 - overlap) * comm_time_s / compute_time_s
    else:
        tp_comm_factor = 0.0

    return tp_comm_factor


def estimate_ttft_ms(variant, hw, tp, quant_bytes, isl, engine_mfu):
    """Estimate TTFT in milliseconds for the prefill phase.

    Calculates the theoretical minimum time to process the input prompt,
    then adjusts for chunked prefill overhead and TP communication.

    Theoretical (compute-bound) TTFT:
        ttft_theoretical_ms = (prefill_flops / (peak_tflops * 1e12 * TP * mfu)) * 1000

    We also check whether the workload is actually compute-bound or
    memory-bandwidth-bound by comparing the arithmetic intensity against
    the hardware's machine balance point.

    Memory-bound TTFT (if applicable):
        ttft_memory_ms = (model_bytes / (hbm_bw * 1e9 * TP)) * 1000

    Engine-adjusted TTFT:
        ttft_adjusted = ttft_theoretical * (1 + chunked_prefill_overhead) * (1 + tp_comm_factor)

    The chunked_prefill_overhead accounts for scheduling overhead when
    chunked prefill is enabled (typically 5-10% for vLLM/SGLang). We use
    a fixed 0.05 (5%) as a conservative default.

    Args:
        variant: Model variant dict (has active_params_B, total_params_B,
                 layers, d, etc.).
        hw: Hardware dict (has fp8_tflops, peak_tflops_bf16, hbm_bw_gbps).
        tp: Tensor parallelism degree.
        quant_bytes: Bytes per param (1=FP8, 2=BF16).
        isl: Input sequence length.
        engine_mfu: Dict with prefill_mfu, tp_comm_overlap, etc.
                    from ENGINE_MFU.

    Returns:
        Dict with:
        - prefill_flops: Total FLOPs for prefilling ISL tokens.
        - peak_tflops_used: Peak TFLOPS selected (FP8 or BF16).
        - mfu: Prefill MFU used for the estimate.
        - ttft_theoretical_ms: Compute-bound TTFT lower bound.
        - ttft_memory_ms: Memory-bound TTFT (for reference).
        - ttft_adjusted_ms: Engine-adjusted TTFT estimate.
        - is_compute_bound: True if prefill is compute-bound.
        - bottleneck: "compute" or "memory".
        - tp_comm_factor: TP communication overhead factor.
        - arithmetic_intensity: Ops per byte.
        - machine_balance: Hardware balance point (ops/byte).
    """
    # FLOPs and peak TFLOPS selection
    prefill_flops = calculate_prefill_flops(variant, isl)
    peak_tflops = _select_peak_tflops(hw, quant_bytes)
    mfu = engine_mfu.get("prefill_mfu", 0.20)

    # Arithmetic intensity check
    ai_info = calculate_arithmetic_intensity(variant, isl, quant_bytes, tp)
    hbm_bw_gbps = hw.get("hbm_bw_gbps", 3350)
    machine_balance = (peak_tflops * 1e12) / (hbm_bw_gbps * 1e9)
    is_compute_bound = ai_info["ops_per_byte"] > machine_balance

    # Theoretical compute-bound TTFT
    effective_tflops = peak_tflops * tp * mfu
    if effective_tflops > 0:
        ttft_theoretical_ms = (prefill_flops / (effective_tflops * 1e12)) * 1000.0
    else:
        ttft_theoretical_ms = float("inf")

    # Memory-bound TTFT (model weight load time)
    total_params = variant["total_params_B"] * 1e9
    model_bytes = total_params * quant_bytes
    model_bytes_per_gpu = model_bytes / tp
    if hbm_bw_gbps > 0:
        ttft_memory_ms = (model_bytes_per_gpu / (hbm_bw_gbps * 1e9)) * 1000.0
    else:
        ttft_memory_ms = float("inf")

    # Pick the binding constraint
    if is_compute_bound:
        ttft_base_ms = ttft_theoretical_ms
        bottleneck = "compute"
    else:
        ttft_base_ms = ttft_memory_ms
        bottleneck = "memory"

    # TP communication overhead
    tp_comm_factor = _calculate_tp_comm_factor(variant, hw, tp, isl, quant_bytes, engine_mfu)

    # Chunked prefill overhead (conservative 5%)
    chunked_prefill_overhead = 0.05

    # Adjusted TTFT
    ttft_adjusted_ms = ttft_base_ms * (1.0 + chunked_prefill_overhead) * (1.0 + tp_comm_factor)

    return {
        "isl": isl,
        "prefill_flops": prefill_flops,
        "peak_tflops_used": peak_tflops,
        "mfu": mfu,
        "ttft_theoretical_ms": round(ttft_theoretical_ms, 3),
        "ttft_memory_ms": round(ttft_memory_ms, 3),
        "ttft_adjusted_ms": round(ttft_adjusted_ms, 3),
        "is_compute_bound": is_compute_bound,
        "bottleneck": bottleneck,
        "tp_comm_factor": round(tp_comm_factor, 4),
        "arithmetic_intensity": round(ai_info["ops_per_byte"], 2),
        "machine_balance": round(machine_balance, 2),
        "model_bytes_per_gpu": model_bytes_per_gpu,
    }


def analyze_prefill(variant, hw, tp, quant_bytes, input_seq_lens, engine_mfu):
    """Analyze prefill for all ISL values.

    Runs estimate_ttft_ms for every ISL in the sweep list and aggregates
    the results with variant/hardware metadata.

    Args:
        variant: Model variant dict.
        hw: Hardware config dict.
        tp: Tensor parallelism degree.
        quant_bytes: Bytes per parameter (1 for FP8, 2 for BF16).
        input_seq_lens: List of ISL values to analyze.
        engine_mfu: Engine MFU dict (prefill_mfu, tp_comm_overlap, etc.).

    Returns:
        Dict with:
        - variant_name: Model name string.
        - hw_name: Hardware platform name.
        - tp: Tensor parallelism degree.
        - quant_bytes: Quantization bytes used.
        - active_params_B: Active parameters in billions.
        - total_params_B: Total parameters in billions.
        - peak_tflops_used: Peak TFLOPS selected for this analysis.
        - isl_results: List of per-ISL result dicts from estimate_ttft_ms.
    """
    peak_tflops = _select_peak_tflops(hw, quant_bytes)

    isl_results = []
    for isl in input_seq_lens:
        result = estimate_ttft_ms(variant, hw, tp, quant_bytes, isl, engine_mfu)
        isl_results.append(result)

    return {
        "variant_name": variant["name"],
        "hw_name": hw["name"],
        "tp": tp,
        "quant_bytes": quant_bytes,
        "active_params_B": variant["active_params_B"],
        "total_params_B": variant["total_params_B"],
        "peak_tflops_used": peak_tflops,
        "isl_results": isl_results,
    }


def generate_prefill_recommendations(all_results):
    """Generate recommendations based on prefill analysis.

    Examines TTFT values and bottleneck types across all hardware/variant
    combinations and produces actionable guidance.

    Args:
        all_results: Dict mapping hw_name -> list of variant prefill analyses
                     (each from analyze_prefill).

    Returns:
        List of recommendation strings.
    """
    recs = []

    for hw_name, variant_results in all_results.items():
        for vr in variant_results:
            variant_name = vr["variant_name"]

            for isl_r in vr["isl_results"]:
                isl = isl_r["isl"]
                ttft = isl_r["ttft_adjusted_ms"]
                bottleneck = isl_r["bottleneck"]

                # Flag high TTFT (> 5 seconds)
                if ttft > 5000:
                    recs.append(
                        f"High TTFT: {variant_name} on {hw_name} at ISL={isl} "
                        f"has TTFT={ttft:.0f} ms ({ttft / 1000:.1f}s). "
                        f"Consider: increase TP, use FP8 quantization, or "
                        f"reduce ISL with chunked prefill."
                    )
                # Flag moderate TTFT (> 2 seconds)
                elif ttft > 2000:
                    recs.append(
                        f"Moderate TTFT: {variant_name} on {hw_name} at ISL={isl} "
                        f"has TTFT={ttft:.0f} ms. Acceptable for batch workloads "
                        f"but may not meet realtime SLAs."
                    )
                # Good TTFT (< 500 ms)
                elif ttft < 500:
                    recs.append(
                        f"Good: {variant_name} on {hw_name} at ISL={isl} "
                        f"TTFT={ttft:.0f} ms — suitable for realtime serving."
                    )

                # Memory-bound warning at short ISL
                if bottleneck == "memory" and isl <= 512:
                    recs.append(
                        f"Memory-bound at short ISL: {variant_name} on {hw_name} "
                        f"at ISL={isl} is memory-bandwidth limited "
                        f"(arithmetic intensity={isl_r['arithmetic_intensity']:.1f} "
                        f"< machine balance={isl_r['machine_balance']:.1f}). "
                        f"Small prompts don't saturate compute; this is normal."
                    )

            # TP communication note
            if vr["tp"] > 1:
                max_comm = max(r["tp_comm_factor"] for r in vr["isl_results"])
                if max_comm > 0.10:
                    recs.append(
                        f"TP overhead: {variant_name} on {hw_name} with TP={vr['tp']} "
                        f"incurs {max_comm * 100:.0f}% communication overhead. "
                        f"Consider a GPU with more memory to reduce TP."
                    )

    return recs


def export_prefill_results_csv(all_results, filename):
    """Export prefill analysis results to CSV.

    Writes one row per (hardware, variant, ISL) combination with all
    prefill metrics: TTFT, bottleneck, arithmetic intensity, etc.

    Args:
        all_results: Dict mapping hw_name -> list of variant prefill analyses.
        filename: Output CSV file path.
    """
    rows = []
    for hw_name, variant_results in all_results.items():
        for vr in variant_results:
            for isl_r in vr["isl_results"]:
                rows.append({
                    "hardware": hw_name,
                    "variant": vr["variant_name"],
                    "tp": vr["tp"],
                    "quant_bytes": vr["quant_bytes"],
                    "active_params_B": vr["active_params_B"],
                    "total_params_B": vr["total_params_B"],
                    "isl": isl_r["isl"],
                    "prefill_flops": isl_r["prefill_flops"],
                    "peak_tflops_used": isl_r["peak_tflops_used"],
                    "mfu": isl_r["mfu"],
                    "ttft_theoretical_ms": isl_r["ttft_theoretical_ms"],
                    "ttft_memory_ms": isl_r["ttft_memory_ms"],
                    "ttft_adjusted_ms": isl_r["ttft_adjusted_ms"],
                    "is_compute_bound": isl_r["is_compute_bound"],
                    "bottleneck": isl_r["bottleneck"],
                    "arithmetic_intensity": isl_r["arithmetic_intensity"],
                    "machine_balance": isl_r["machine_balance"],
                    "tp_comm_factor": isl_r["tp_comm_factor"],
                })

    if rows:
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
