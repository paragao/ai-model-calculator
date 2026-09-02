"""
Phase 10: Tensor Parallel Communication Analysis for Inference.

Analyzes TP communication overhead during inference decode steps.
TP splits model weights across GPUs; each transformer layer requires
2 all-reduce operations (attention output + MLP output) per decode step.

Key formulas:
  # Volume per all-reduce (ring all-reduce):
  #   2 * (TP - 1) / TP * message_size
  # Message size per layer = hidden_dim * batch * dtype_bytes

  # Per-layer comm volume (2 all-reduces: attn + MLP):
  comm_per_layer = 2 * 2 * (tp - 1) / tp * hidden_dim * batch * dtype_bytes

  # Total per decode step:
  total_comm_bytes = comm_per_layer * layers

  # Comm time:
  #   NVLink (intra_node_bw_gbps > 200): use intra_node_bw_gbps
  #   PCIe  (intra_node_bw_gbps <= 200): use pcie_bw_gbps * allreduce_factor
  #   If TP > gpus_per_node: inter-node via EFA (inter_node_bw_gb)
  comm_time_per_step = total_comm_bytes / (interconnect_bw * 1e9)

  # TP efficiency:
  tp_efficiency = compute_time / (compute_time + comm_time * (1 - overlap_fraction))

Notes:
  - Detect NVLink vs PCIe by intra_node_bw_gbps threshold (>200 = NVLink).
  - If TP > gpus_per_node, communication goes inter-node via EFA.
  - Overlap fraction from engine_mfu["tp_comm_overlap"].

References:
  - NCCL ring all-reduce algorithm: each operation exchanges
    2 * (N-1)/N * message_size bytes in 2*(N-1) steps across N ranks.
    https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
  - PCIe effective bandwidth factors measured via:
    nccl-tests all_reduce_perf -b 8 -e 256M -f 2 -g <num_gpus>
    https://github.com/NVIDIA/nccl-tests
  - Launch latency = NCCL kernel dispatch + inter-GPU synchronization,
    measured from nccl-tests with small message sizes (8B-4KB).
  - PCIe factors and launch latencies are configurable per instance type
    in hardware_config.py INFERENCE_HARDWARE entries.
"""

import math
import csv


def _detect_interconnect(hw, tp):
    """Detect interconnect type and effective bandwidth for TP communication.

    Logic:
      - If TP > gpus_per_node: inter-node via EFA (inter_node_bw_gb).
      - If intra_node_bw_gbps > 200: NVLink (P-family, high-BW).
      - Otherwise: PCIe (G-family, low-BW).

    For PCIe, the effective all-reduce bandwidth is significantly lower
    than the per-link peak due to multi-hop ring topology, dual-socket
    CPU architectures, and NCCL implementation overhead. The bandwidth
    scaling factors are read from the hardware config
    (pcie_allreduce_factor_{2,4,8}gpu), which are empirical measurements
    from nccl-tests all_reduce_perf on each instance type.

    Launch latencies per all-reduce operation are also read from hardware
    config (launch_latency_us_{nvlink,pcie,efa}). These represent NCCL
    kernel dispatch + synchronization overhead, measured via nccl-tests
    with small message sizes where latency dominates.

    See CONSTANTS.md for measurement methodology and source references.

    Args:
        hw: Hardware config dict.
        tp: Tensor parallelism degree.

    Returns:
        Tuple of (effective_bandwidth_gbps, interconnect_type_string,
                  launch_latency_us).
    """
    gpus_per_node = hw.get("gpus_per_node", 8)
    intra_bw = hw.get("intra_node_bw_gbps", 900)
    pcie_bw = hw.get("pcie_bw_gbps", 64)
    inter_bw = hw.get("inter_node_bw_gb", 400)

    if tp > gpus_per_node:
        latency = hw.get("launch_latency_us_efa", 75.0)
        return (inter_bw, "EFA", latency)
    elif intra_bw > 200:
        latency = hw.get("launch_latency_us_nvlink", 8.0)
        return (intra_bw, "NVLink", latency)
    else:
        # PCIe effective bandwidth for ring all-reduce with TP GPUs.
        # Read empirical NCCL bus bandwidth factors from hardware config
        # (with fallback defaults matching g6e measurements).
        # Source: nccl-tests all_reduce_perf -b 8 -e 256M -f 2 -g <TP>
        factor_2gpu = hw.get("pcie_allreduce_factor_2gpu", 0.60)
        factor_4gpu = hw.get("pcie_allreduce_factor_4gpu", 0.15)
        factor_8gpu = hw.get("pcie_allreduce_factor_8gpu", 0.05)

        if tp <= 2:
            pcie_factor = factor_2gpu
        elif tp <= 4:
            pcie_factor = factor_4gpu
        else:
            pcie_factor = factor_8gpu

        latency = hw.get("launch_latency_us_pcie", 40.0)
        return (pcie_bw * pcie_factor, "PCIe", latency)


def calculate_tp_comm_volume(variant, tp, batch_size, dtype_bytes):
    """Total bytes communicated per decode step across all layers.

    Each transformer layer performs 2 all-reduce operations during decode
    (one after attention projection, one after MLP). Ring all-reduce for
    TP GPUs exchanges 2 * (TP-1)/TP * message_size bytes per operation.

    Formula:
        message_size = hidden_dim * batch_size * dtype_bytes
        comm_per_layer = 2 * 2 * (tp - 1) / tp * message_size
        total_bytes = comm_per_layer * layers

    Args:
        variant: Model variant dict (has d, layers).
        tp: Tensor parallelism degree.
        batch_size: Number of concurrent requests (tokens generated per step).
        dtype_bytes: Bytes per activation element (2 for BF16, 1 for FP8).

    Returns:
        Dict with:
        - comm_per_layer_bytes: Communication volume per layer in bytes.
        - total_comm_bytes: Total communication volume per decode step.
        - message_size_bytes: Message size per all-reduce operation.
        - num_allreduces_per_layer: Number of all-reduce ops per layer (2).
        - layers: Total layers in model.
    """
    if tp <= 1:
        return {
            "comm_per_layer_bytes": 0,
            "total_comm_bytes": 0,
            "message_size_bytes": 0,
            "num_allreduces_per_layer": 2,
            "layers": variant["layers"],
        }

    hidden_dim = variant["d"]
    layers = variant["layers"]

    # Message size per all-reduce: hidden_dim * batch_size * dtype_bytes
    message_size = hidden_dim * batch_size * dtype_bytes

    # Ring all-reduce exchanges 2 * (TP-1)/TP * message_size per operation
    # 2 all-reduces per layer (attention output + MLP output)
    allreduce_volume = 2.0 * (tp - 1) / tp * message_size
    comm_per_layer = 2 * allreduce_volume  # 2 all-reduces per layer

    total_comm_bytes = comm_per_layer * layers

    return {
        "comm_per_layer_bytes": comm_per_layer,
        "total_comm_bytes": total_comm_bytes,
        "message_size_bytes": message_size,
        "num_allreduces_per_layer": 2,
        "layers": layers,
    }


def estimate_tp_comm_time_ms(variant, hw, tp, batch_size, dtype_bytes):
    """Communication time per decode step in milliseconds.

    Uses NVLink (intra_node_bw_gbps) for high-bandwidth interconnects
    (>200 GB/s), PCIe (pcie_bw_gbps) otherwise. If TP > gpus_per_node,
    uses inter-node bandwidth (inter_node_bw_gb) via EFA.

    Formula:
        comm_time_ms = total_comm_bytes / (interconnect_bw * 1e9) * 1000

    Args:
        variant: Model variant dict.
        hw: Hardware config dict (has intra_node_bw_gbps, pcie_bw_gbps,
            inter_node_bw_gb, gpus_per_node).
        tp: Tensor parallelism degree.
        batch_size: Number of concurrent requests.
        dtype_bytes: Bytes per activation element.

    Returns:
        Dict with:
        - comm_time_ms: Communication time in milliseconds.
        - interconnect_bw_gbps: Bandwidth used for calculation.
        - interconnect_type: "NVLink", "PCIe", or "EFA".
        - total_comm_bytes: Total bytes communicated.
    """
    vol = calculate_tp_comm_volume(variant, tp, batch_size, dtype_bytes)
    total_bytes = vol["total_comm_bytes"]

    if total_bytes == 0 or tp <= 1:
        return {
            "comm_time_ms": 0.0,
            "interconnect_bw_gbps": 0.0,
            "interconnect_type": "none",
            "total_comm_bytes": 0,
        }

    bw_gbps, ic_type, launch_latency_us = _detect_interconnect(hw, tp)

    # Number of all-reduce operations per decode step
    # 2 per layer (attention + MLP) x num_layers
    layers = variant["layers"]
    num_allreduces = 2 * layers

    # NCCL kernel launch latency per all-reduce operation.
    # Read from hardware config via _detect_interconnect().
    # Source: nccl-tests all_reduce_perf with small message sizes (8B-4KB)
    # where latency dominates over bandwidth.
    # Typical values:
    #   NVLink: ~5-10 us (fast intra-node, minimal synchronization)
    #   PCIe:   ~30-50 us (ring all-reduce over PCIe bus, CPU-mediated sync)
    #   EFA:    ~50-100 us (network RTT + NCCL proxy thread overhead)

    # Total launch latency for all operations
    # Convert microseconds to milliseconds (1000 us/ms)
    launch_latency_ms = num_allreduces * launch_latency_us / 1000.0

    if bw_gbps > 0:
        # Bandwidth component: total bytes / effective bandwidth
        comm_time_s = total_bytes / (bw_gbps * 1e9)
        # Convert seconds to milliseconds (1000 ms/s)
        bw_time_ms = comm_time_s * 1000.0
        # Total = bandwidth time + launch latency
        comm_time_ms = bw_time_ms + launch_latency_ms
    else:
        comm_time_ms = float("inf")

    return {
        "comm_time_ms": round(comm_time_ms, 4),
        "interconnect_bw_gbps": bw_gbps,
        "interconnect_type": ic_type,
        "total_comm_bytes": total_bytes,
        "launch_latency_ms": round(launch_latency_ms, 4),
    }


def calculate_tp_efficiency(variant, hw, tp, batch_size, quant_bytes,
                            dtype_bytes, engine_mfu):
    """TP efficiency factor (0-1) comparing compute vs communication time.

    Compute time is estimated from model FLOPs and peak TFLOPS.
    Communication time comes from estimate_tp_comm_time_ms.
    Overlap fraction from engine_mfu reduces effective comm time.

    Formulas:
        flops_per_step = 2 * active_params_B * 1e9 * batch_size
        compute_time_ms = flops_per_step / (peak_tflops * 1e12 * tp) * 1000
        effective_comm_ms = comm_time_ms * (1 - overlap_fraction)
        efficiency = compute_time_ms / (compute_time_ms + effective_comm_ms)

    Args:
        variant: Model variant dict (has active_params_B).
        hw: Hardware config dict (has peak_tflops_bf16, fp8_tflops).
        tp: Tensor parallelism degree.
        batch_size: Number of concurrent requests.
        quant_bytes: Bytes per model parameter (1=FP8, 2=BF16).
        dtype_bytes: Bytes per activation element.
        engine_mfu: Engine MFU dict (has tp_comm_overlap).

    Returns:
        Dict with:
        - compute_time_ms: Compute time per decode step.
        - comm_time_ms: Raw communication time.
        - effective_comm_ms: Comm time after overlap.
        - overlap_fraction: Fraction of comm overlapped with compute.
        - efficiency: TP efficiency factor (0-1).
        - interconnect_type: "NVLink", "PCIe", or "EFA".
        - bottleneck: "compute" or "communication".
    """
    if tp <= 1:
        return {
            "compute_time_ms": 0.0,
            "comm_time_ms": 0.0,
            "effective_comm_ms": 0.0,
            "overlap_fraction": 0.0,
            "efficiency": 1.0,
            "interconnect_type": "none",
            "bottleneck": "compute",
        }

    # Select peak TFLOPS based on quantization
    if quant_bytes <= 1:
        fp8 = hw.get("fp8_tflops", 0)
        peak_tflops = float(fp8) if fp8 > 0 else float(hw["peak_tflops_bf16"])
    else:
        peak_tflops = float(hw["peak_tflops_bf16"])

    # Compute time: FLOPs per decode step / (peak compute * TP)
    # Each token: 2 * active_params FLOPs; batch_size tokens per step
    active_params = variant["active_params_B"] * 1e9
    flops_per_step = 2.0 * active_params * batch_size
    if peak_tflops > 0 and tp > 0:
        compute_time_s = flops_per_step / (peak_tflops * 1e12 * tp)
        # Convert seconds to milliseconds (1000 ms/s)
        compute_time_ms = compute_time_s * 1000.0
    else:
        compute_time_ms = float("inf")

    # Communication time
    comm_info = estimate_tp_comm_time_ms(variant, hw, tp, batch_size, dtype_bytes)
    comm_time_ms = comm_info["comm_time_ms"]
    ic_type = comm_info["interconnect_type"]

    # Overlap fraction
    overlap = engine_mfu.get("tp_comm_overlap", 0.0)
    effective_comm_ms = comm_time_ms * (1.0 - overlap)

    # TP efficiency: fraction of time spent on useful compute
    total_time = compute_time_ms + effective_comm_ms
    if total_time > 0:
        efficiency = compute_time_ms / total_time
    else:
        efficiency = 1.0

    bottleneck = "communication" if effective_comm_ms > compute_time_ms else "compute"

    return {
        "compute_time_ms": round(compute_time_ms, 4),
        "comm_time_ms": round(comm_time_ms, 4),
        "effective_comm_ms": round(effective_comm_ms, 4),
        "overlap_fraction": overlap,
        "efficiency": round(efficiency, 4),
        "interconnect_type": ic_type,
        "bottleneck": bottleneck,
    }


def analyze_tp_comm(variant, hw_list, tp_values, batch_sizes,
                    quant_bytes, dtype_bytes, engine_mfu):
    """Sweep TP and batch sizes across hardware configurations.

    For each (hardware, tp, batch_size) combination, calculates comm volume,
    comm time, and TP efficiency. Skips configurations where TP=1 (no comm).

    Args:
        variant: Model variant dict.
        hw_list: List of hardware config dicts.
        tp_values: List of TP degrees to sweep.
        batch_sizes: List of batch sizes to sweep.
        quant_bytes: Bytes per model parameter.
        dtype_bytes: Bytes per activation element.
        engine_mfu: Engine MFU dict.

    Returns:
        Dict with:
        - variant_name: Model name.
        - results: Dict mapping hw_name -> list of per-(tp, batch) result dicts.
    """
    results = {}

    for hw in hw_list:
        hw_name = hw["name"]
        hw_results = []

        for tp in tp_values:
            if tp <= 1:
                continue  # No communication at TP=1

            for bs in batch_sizes:
                vol = calculate_tp_comm_volume(variant, tp, bs, dtype_bytes)
                comm = estimate_tp_comm_time_ms(variant, hw, tp, bs, dtype_bytes)
                eff = calculate_tp_efficiency(
                    variant, hw, tp, bs, quant_bytes, dtype_bytes, engine_mfu
                )

                hw_results.append({
                    "tp": tp,
                    "batch_size": bs,
                    "total_comm_bytes": vol["total_comm_bytes"],
                    "comm_per_layer_bytes": vol["comm_per_layer_bytes"],
                    "comm_time_ms": comm["comm_time_ms"],
                    "interconnect_type": comm["interconnect_type"],
                    "interconnect_bw_gbps": comm["interconnect_bw_gbps"],
                    "compute_time_ms": eff["compute_time_ms"],
                    "effective_comm_ms": eff["effective_comm_ms"],
                    "efficiency": eff["efficiency"],
                    "bottleneck": eff["bottleneck"],
                    "overlap_fraction": eff["overlap_fraction"],
                })

        results[hw_name] = hw_results

    return {
        "variant_name": variant["name"],
        "results": results,
    }


def generate_tp_recommendations(all_results):
    """Generate recommendations based on TP communication analysis.

    Args:
        all_results: Dict from analyze_tp_comm.

    Returns:
        List of recommendation strings.
    """
    recs = []
    variant_name = all_results.get("variant_name", "Unknown")

    for hw_name, hw_results in all_results.get("results", {}).items():
        if not hw_results:
            continue

        # Track efficiency ranges
        min_eff = 1.0
        max_eff = 0.0
        comm_bound_count = 0
        pcie_configs = 0
        efa_configs = 0

        for r in hw_results:
            eff = r["efficiency"]
            min_eff = min(min_eff, eff)
            max_eff = max(max_eff, eff)
            if r["bottleneck"] == "communication":
                comm_bound_count += 1
            if r["interconnect_type"] == "PCIe":
                pcie_configs += 1
            if r["interconnect_type"] == "EFA":
                efa_configs += 1

        total = len(hw_results)

        # Flag low TP efficiency
        if min_eff < 0.50:
            worst = min(hw_results, key=lambda x: x["efficiency"])
            recs.append(
                f"Low TP efficiency: {variant_name} on {hw_name} with "
                f"TP={worst['tp']}, batch={worst['batch_size']} has "
                f"efficiency={worst['efficiency']:.0%}. "
                f"Communication ({worst['interconnect_type']}) dominates. "
                f"Consider reducing TP or using hardware with higher "
                f"interconnect bandwidth."
            )

        # Flag PCIe bottleneck
        if pcie_configs > 0:
            recs.append(
                f"PCIe interconnect: {variant_name} on {hw_name} uses PCIe "
                f"for {pcie_configs}/{total} TP configs. PCIe has ~10-15x "
                f"lower bandwidth than NVLink. Consider P-family instances "
                f"for better TP scaling."
            )

        # Flag inter-node TP
        if efa_configs > 0:
            recs.append(
                f"Inter-node TP: {variant_name} on {hw_name} requires "
                f"inter-node (EFA) communication for {efa_configs}/{total} "
                f"configs. TP across nodes has much higher latency. "
                f"Consider keeping TP within a single node."
            )

        # Good efficiency
        if min_eff >= 0.85 and total > 0:
            recs.append(
                f"Good TP efficiency: {variant_name} on {hw_name} achieves "
                f"{min_eff:.0%}-{max_eff:.0%} efficiency across all TP/batch "
                f"configurations."
            )

        # Communication-bound at high batch
        if comm_bound_count > 0:
            recs.append(
                f"Communication-bound: {variant_name} on {hw_name} is "
                f"communication-bound in {comm_bound_count}/{total} configs. "
                f"Increasing batch size can amortize comm overhead."
            )

    return recs


def export_tp_results_csv(all_results, filename):
    """Export TP communication analysis results to CSV.

    Args:
        all_results: Dict from analyze_tp_comm.
        filename: Output CSV file path.
    """
    rows = []
    variant_name = all_results.get("variant_name", "Unknown")

    for hw_name, hw_results in all_results.get("results", {}).items():
        for r in hw_results:
            rows.append({
                "variant": variant_name,
                "hardware": hw_name,
                "tp": r["tp"],
                "batch_size": r["batch_size"],
                "total_comm_bytes": r["total_comm_bytes"],
                "comm_per_layer_bytes": r["comm_per_layer_bytes"],
                "comm_time_ms": r["comm_time_ms"],
                "interconnect_type": r["interconnect_type"],
                "interconnect_bw_gbps": r["interconnect_bw_gbps"],
                "compute_time_ms": r["compute_time_ms"],
                "effective_comm_ms": r["effective_comm_ms"],
                "efficiency": r["efficiency"],
                "bottleneck": r["bottleneck"],
                "overlap_fraction": r["overlap_fraction"],
            })

    if rows:
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
