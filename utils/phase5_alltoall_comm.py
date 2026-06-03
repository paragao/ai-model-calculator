"""
Phase 5: All-to-All Communication Analysis functions.

Analyzes intra-node all-to-all communication volumes for MoE expert routing
across different micro batch sizes.
"""

# Import configurations
from configuration.project_config import *
from configuration.advanced_config import *

# Import formatting utilities
from .formatting_utils import color_text


def analyze_alltoall_communication(variant, hw, micro_batch_sizes):
    """
    Analyze all-to-all communication volumes for MoE routing.

    Args:
        variant: Variant dict with model architecture
        hw: Hardware dict with GPU specs
        micro_batch_sizes: List of micro batch sizes to analyze

    Returns:
        dict with:
        - variant_name: Name of variant
        - d: Hidden dimension
        - a2a_metrics: list of dicts with micro-batch-specific metrics
    """
    d = variant["d"]
    v_name = variant["name"]
    intra_node_bw = hw["intra_node_bw_gbps"]

    # Calculate metrics for each micro batch size
    a2a_metrics = []
    for micro in micro_batch_sizes:
        metrics = calculate_alltoall_metrics(micro, d, intra_node_bw)
        metrics["micro"] = micro
        a2a_metrics.append(metrics)

    return {
        "variant_name": v_name,
        "d": d,
        "a2a_metrics": a2a_metrics
    }


def calculate_alltoall_metrics(micro, d, intra_node_bw):
    """
    Calculate all-to-all communication volume and time for MoE routing.

    Args:
        micro: Micro batch size
        d: Hidden dimension
        intra_node_bw: Intra-node bandwidth in GB/s

    Returns:
        dict with:
        - fwd_bwd_gb: Communication volume for forward+backward passes
        - a2a_time_ms: All-to-all communication time in milliseconds
        - performance_rating: Rating (0=excellent, 3=poor)
    """
    # All-to-all for MoE routing: activations sent to experts
    # Formula: SEQ_LEN * micro * TOPK * d * PARAM_BYTES
    fwd_bytes = SEQ_LEN * micro * TOPK * d * PARAM_BYTES
    fwd_bwd_gb = fwd_bytes * 2 / 1e9  # Factor of 2 for forward + backward passes

    # Calculate communication time at intra-node bandwidth
    a2a_time_ms = fwd_bwd_gb / intra_node_bw * 1000

    # Assign performance rating based on communication time
    # These thresholds are heuristics for MoE all-to-all overhead
    if a2a_time_ms < 1.0:
        performance_rating = 0  # Excellent (<1ms)
    elif a2a_time_ms < 3.0:
        performance_rating = 1  # Good (1-3ms)
    elif a2a_time_ms < 5.0:
        performance_rating = 2  # Moderate (3-5ms)
    else:
        performance_rating = 3  # High (>5ms)

    return {
        "fwd_bwd_gb": fwd_bwd_gb,
        "a2a_time_ms": a2a_time_ms,
        "performance_rating": performance_rating
    }


def generate_alltoall_recommendations(a2a_results):
    """
    Analyze all-to-all communication results and generate recommendations.

    Args:
        a2a_results: dict with all-to-all analysis results

    Returns:
        list of recommendation strings
    """
    recommendations = []

    if not a2a_results.get("hardware"):
        return ["⚠️  No all-to-all communication analysis available"]

    # Analyze overall communication patterns
    total_configs = 0
    excellent_configs = 0
    high_overhead_configs = 0
    max_time = 0
    max_config = None

    for hw_name, hw_data in a2a_results["hardware"].items():
        for variant_data in hw_data:
            for metrics in variant_data["a2a_metrics"]:
                total_configs += 1
                rating = metrics["performance_rating"]

                if rating == 0:
                    excellent_configs += 1
                elif rating == 3:
                    high_overhead_configs += 1

                # Track maximum communication time
                if metrics["a2a_time_ms"] > max_time:
                    max_time = metrics["a2a_time_ms"]
                    max_config = (hw_name, variant_data["variant_name"], metrics["micro"])

    if total_configs > 0:
        excellent_pct = (excellent_configs / total_configs) * 100
        high_pct = (high_overhead_configs / total_configs) * 100

        if excellent_pct >= 70:
            recommendations.append(
                f"✅ All-to-all communication is excellent for {excellent_configs}/{total_configs} configs (<1ms)"
            )
        elif high_pct >= 30:
            recommendations.append(
                f"⚠️  All-to-all communication overhead is high for {high_overhead_configs}/{total_configs} configs (>5ms)"
            )

    # Identify bottleneck configuration
    if max_config and max_time > 5.0:
        hw_name, variant_name, micro = max_config
        recommendations.append(
            f"🔍 Highest overhead: Variant {variant_name} on {hw_name} with micro={micro} ({max_time:.1f}ms) "
            f"→ Consider reducing micro batch size or improving intra-node bandwidth"
        )

    # Analyze micro batch size impact
    micro_impacts = {}
    for hw_name, hw_data in a2a_results["hardware"].items():
        for variant_data in hw_data:
            for metrics in variant_data["a2a_metrics"]:
                micro = metrics["micro"]
                if micro not in micro_impacts:
                    micro_impacts[micro] = []
                micro_impacts[micro].append(metrics["a2a_time_ms"])

    if len(micro_impacts) > 1:
        avg_times = {m: sum(times) / len(times) for m, times in micro_impacts.items()}
        min_micro = min(avg_times, key=avg_times.get)
        max_micro = max(avg_times, key=avg_times.get)

        if avg_times[max_micro] > 2 * avg_times[min_micro]:
            recommendations.append(
                f"📊 Micro batch size impact: Larger batches increase all-to-all overhead "
                f"(micro={max_micro}: {avg_times[max_micro]:.1f}ms avg vs micro={min_micro}: {avg_times[min_micro]:.1f}ms avg)"
            )

    # Hardware bandwidth recommendations
    hw_bandwidth_map = {}
    for hw_name, hw_data in a2a_results["hardware"].items():
        avg_time = sum(
            m["a2a_time_ms"]
            for vd in hw_data
            for m in vd["a2a_metrics"]
        ) / sum(len(vd["a2a_metrics"]) for vd in hw_data)
        hw_bandwidth_map[hw_name] = avg_time

    if hw_bandwidth_map:
        best_hw = min(hw_bandwidth_map, key=hw_bandwidth_map.get)
        worst_hw = max(hw_bandwidth_map, key=hw_bandwidth_map.get)

        if hw_bandwidth_map[worst_hw] > 1.5 * hw_bandwidth_map[best_hw]:
            recommendations.append(
                f"🚀 Hardware choice matters: {best_hw} has {hw_bandwidth_map[worst_hw] / hw_bandwidth_map[best_hw]:.1f}x "
                f"faster all-to-all than {worst_hw} (better intra-node bandwidth)"
            )

    return recommendations


def calculate_moe_all_to_all(layers_per_gpu, hidden_dim, seq_len, mbs, gbs, dp, ep,
                             topk, num_experts, dtype_bytes=2,
                             intra_node_bw_gbps=1800, inter_node_bw_gb=400,
                             gpus_per_node=8):
    """
    Comprehensive MoE All-to-All communication analysis.

    Args:
        layers_per_gpu: MoE layers on this GPU
        hidden_dim: Model hidden dimension
        seq_len: Sequence length
        mbs: Micro batch size
        gbs: Global batch size (in sequences)
        dp: Data parallelism degree
        ep: Expert parallelism degree
        topk: Number of experts selected per token
        num_experts: Total number of experts
        dtype_bytes: Bytes per element (2 for BF16)
        intra_node_bw_gbps: Intra-node bandwidth in GB/s
        inter_node_bw_gb: Inter-node bandwidth in GB/s
        gpus_per_node: GPUs per node

    Returns:
        dict with per-layer, per-step, and bandwidth metrics
    """
    # Per-layer A2A volume: each token sends hidden_dim to topk experts
    # Forward: tokens dispatched to experts + gathered back
    a2a_per_layer_fwd_bytes = seq_len * mbs * topk * hidden_dim * dtype_bytes
    a2a_per_layer_bwd_bytes = a2a_per_layer_fwd_bytes  # symmetric
    a2a_per_layer_bytes = a2a_per_layer_fwd_bytes + a2a_per_layer_bwd_bytes

    # Per micro-batch: across all MoE layers on this GPU
    a2a_per_microbatch_bytes = layers_per_gpu * a2a_per_layer_bytes

    # Number of micro-batches per training step
    num_microbatches = gbs // (mbs * dp) if (mbs * dp) > 0 else 1

    # Total A2A per training step
    total_a2a_bytes = a2a_per_microbatch_bytes * num_microbatches

    # Determine if A2A is intra-node or inter-node
    # EP GPUs within same node use NVLink; cross-node uses EFA
    is_intra_node = ep <= gpus_per_node
    effective_bw = intra_node_bw_gbps if is_intra_node else inter_node_bw_gb
    comm_type = "intra-node (NVLink)" if is_intra_node else "inter-node (EFA)"

    # A2A bandwidth: each GPU sends (EP-1)/EP fraction of its data
    # Ring-based A2A effective BW = link_bw * (EP-1)/EP for large messages
    ep_efficiency = (ep - 1) / ep if ep > 1 else 1.0
    effective_a2a_bw = effective_bw * ep_efficiency

    # Timing estimates
    a2a_per_layer_ms = (a2a_per_layer_bytes / 1e9) / effective_a2a_bw * 1000 if effective_a2a_bw > 0 else 0
    a2a_per_microbatch_ms = a2a_per_layer_ms * layers_per_gpu
    a2a_per_step_ms = a2a_per_microbatch_ms * num_microbatches

    # Bandwidth utilization
    peak_bw = intra_node_bw_gbps if is_intra_node else inter_node_bw_gb
    achieved_bw_gbps = (a2a_per_layer_bytes / 1e9) / (a2a_per_layer_ms / 1000) if a2a_per_layer_ms > 0 else 0
    bw_utilization_pct = (achieved_bw_gbps / peak_bw * 100) if peak_bw > 0 else 0

    return {
        "a2a_per_layer_fwd_mb": a2a_per_layer_fwd_bytes / 1e6,
        "a2a_per_layer_total_mb": a2a_per_layer_bytes / 1e6,
        "a2a_per_microbatch_mb": a2a_per_microbatch_bytes / 1e6,
        "num_microbatches": num_microbatches,
        "total_a2a_per_step_gb": total_a2a_bytes / 1e9,
        "comm_type": comm_type,
        "is_intra_node": is_intra_node,
        "effective_bw_gbps": effective_a2a_bw,
        "a2a_per_layer_ms": a2a_per_layer_ms,
        "a2a_per_microbatch_ms": a2a_per_microbatch_ms,
        "a2a_per_step_ms": a2a_per_step_ms,
        "bw_utilization_pct": bw_utilization_pct,
        "layers_per_gpu": layers_per_gpu,
        "ep": ep,
    }


def export_alltoall_results_csv(a2a_results, filename):
    """
    Export all-to-all communication analysis results to CSV file.

    Args:
        a2a_results: dict with all-to-all analysis results
        filename: Output CSV filename
    """
    import csv

    with open(filename, 'w', newline='') as f:
        fieldnames = [
            'hardware', 'variant', 'd', 'micro', 'fwd_bwd_gb',
            'a2a_time_ms', 'performance_rating'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for hw_name, hw_data in a2a_results.get("hardware", {}).items():
            for variant_data in hw_data:
                for metrics in variant_data["a2a_metrics"]:
                    writer.writerow({
                        'hardware': hw_name,
                        'variant': variant_data['variant_name'],
                        'd': variant_data['d'],
                        'micro': metrics['micro'],
                        'fwd_bwd_gb': metrics['fwd_bwd_gb'],
                        'a2a_time_ms': metrics['a2a_time_ms'],
                        'performance_rating': metrics['performance_rating']
                    })
