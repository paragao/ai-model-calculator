"""
Phase 4.1: ZeRO-1 Communication Overhead Analysis functions.

Analyzes all-gather communication overhead for ZeRO-1 optimization strategy
(optimizer states only) across different data parallelism degrees.
"""

# Import configurations
from project_config import *
from advanced_config import *

# Import core calculation functions
from .core_calculations import (
    calculate_model_memory_gb,
    calculate_viable_micro_batch_sizes
)

# Import formatting utilities
from .formatting_utils import color_text


def analyze_zero1_communication_overhead(variant, hw, dp_values, layers_per_gpu, experts_per_gpu):
    """
    Analyze ZeRO-1 all-gather communication overhead for a variant/hardware combination.

    ZeRO-1 shards optimizer states only. Communication happens via all-gather
    AFTER optimizer step to collect optimizer state updates.

    Args:
        variant: Variant dict with model architecture
        hw: Hardware dict with GPU specs
        dp_values: List of DP values to analyze
        layers_per_gpu: Number of layers per GPU
        experts_per_gpu: Number of experts per GPU

    Returns:
        dict with:
        - viable: bool indicating if variant fits with ZeRO-1
        - viable_micros: list of viable micro batch sizes
        - model_mem_gb: model memory per GPU
        - comm_metrics: list of dicts with DP-specific metrics
    """
    gpu_mem = hw["mem_gb"]
    dp_full = hw["gpus"] // (TP * PP * CP)

    # Calculate model memory
    model_mem_gb = calculate_model_memory_gb(
        variant,
        layers_per_gpu,
        experts_per_gpu,
        PARAM_BYTES,
        VOCAB,
        LAYER_NORM,
        FFN_WEIGHT_MATRICES,
        NUM_EMBEDDINGS_TABLES,
    )

    # Check viable micro batch sizes with ZeRO-1
    viable_micros = calculate_viable_micro_batch_sizes(
        variant, gpu_mem, dp_full, layers_per_gpu, experts_per_gpu, zero_strategy=1
    )

    if not viable_micros:
        return {
            "viable": False,
            "viable_micros": [],
            "model_mem_gb": model_mem_gb,
            "comm_metrics": []
        }

    # Calculate communication metrics for each DP value
    comm_metrics = []
    for dp_val in dp_values:
        metrics = calculate_zero1_metrics(
            model_mem_gb, dp_val, hw["inter_node_bw_gb"]
        )
        metrics["dp"] = dp_val
        comm_metrics.append(metrics)

    return {
        "viable": True,
        "viable_micros": viable_micros,
        "model_mem_gb": model_mem_gb,
        "comm_metrics": comm_metrics
    }


def calculate_zero1_metrics(model_mem_gb, dp_val, inter_node_bw_gb):
    """
    Calculate ZeRO-1 all-gather communication volume and time.

    ZeRO-1 communicates optimizer states only (not gradients).
    Formula: volume = model_mem_gb * (OPTIM_BYTES/PARAM_BYTES) / dp_val * 1000

    Args:
        model_mem_gb: Model memory per GPU in GB
        dp_val: Data parallelism degree
        inter_node_bw_gb: Inter-node bandwidth in GB/s

    Returns:
        dict with:
        - ag_volume_mb: All-gather volume in MB per micro-batch
        - ag_time_ms: All-gather time in milliseconds
        - overhead_rating: Performance rating (0=excellent, 3=poor)
    """
    # All-gather volume: optimizer states only
    # OPTIM_BYTES = 6 (momentum + variance + FP32 master)
    # PARAM_BYTES = 2 (BF16)
    # Factor = 6/2 = 3
    ag_volume_mb = model_mem_gb * 3 / dp_val * 1000

    # Communication time at inter-node bandwidth
    ag_time_ms = ag_volume_mb / inter_node_bw_gb

    # Assign overhead rating (adjusted for lower ZeRO-1 overhead)
    # These thresholds are proportionally adjusted from ZeRO-2
    if ag_time_ms < 1.5:
        overhead_rating = 0  # Excellent (<1.5ms)
    elif ag_time_ms < 3.0:
        overhead_rating = 1  # Good (1.5-3ms)
    elif ag_time_ms < 7.0:
        overhead_rating = 2  # Moderate (3-7ms)
    else:
        overhead_rating = 3  # High (>7ms)

    return {
        "ag_volume_mb": ag_volume_mb,
        "ag_time_ms": ag_time_ms,
        "overhead_rating": overhead_rating
    }


def generate_zero1_recommendations(zero1_results):
    """
    Analyze ZeRO-1 communication results and generate recommendations.

    Args:
        zero1_results: dict with all-to-all analysis results

    Returns:
        list of recommendation strings
    """
    recommendations = []

    if not zero1_results.get("hardware"):
        return ["⚠️  No ZeRO-1 communication analysis available"]

    # Analyze overall communication patterns
    total_configs = 0
    excellent_configs = 0
    good_configs = 0
    high_overhead_configs = 0

    for hw_name, hw_data in zero1_results["hardware"].items():
        for variant_name, variant_data in hw_data.items():
            if variant_data["viable"]:
                for metrics in variant_data["comm_metrics"]:
                    total_configs += 1
                    rating = metrics["overhead_rating"]
                    if rating == 0:
                        excellent_configs += 1
                    elif rating == 1:
                        good_configs += 1
                    elif rating == 3:
                        high_overhead_configs += 1

    if total_configs > 0:
        excellent_pct = (excellent_configs / total_configs) * 100
        high_pct = (high_overhead_configs / total_configs) * 100

        if excellent_pct >= 60:
            recommendations.append(
                f"✅ ZeRO-1 communication overhead is excellent for {excellent_configs}/{total_configs} configs (<1.5ms)"
            )
        elif high_pct >= 30:
            recommendations.append(
                f"⚠️  ZeRO-1 communication overhead is high for {high_overhead_configs}/{total_configs} configs (>7ms)"
            )

    # Highlight ZeRO-1 vs ZeRO-2 advantage
    recommendations.append(
        "🚀 ZeRO-1 has ~67% less communication than ZeRO-2 (3x vs 2x factor) "
        "→ Prefer ZeRO-1 when memory permits for 10% efficiency gain"
    )

    # Count variants that fit with ZeRO-1
    viable_variants = []
    for hw_name, hw_data in zero1_results["hardware"].items():
        for variant_name, variant_data in hw_data.items():
            if variant_data["viable"] and variant_name not in viable_variants:
                viable_variants.append(variant_name)

    if viable_variants:
        recommendations.append(
            f"💡 Variants {', '.join(viable_variants)} fit with ZeRO-1: "
            f"Recommended for best performance (efficiency=1.0 vs 0.9 for ZeRO-2)"
        )

    # DP scaling recommendation
    recommendations.append(
        "🎯 Optimal DP configurations: Higher DP = lower per-GPU communication volume "
        "(linear scaling: doubling DP halves communication)"
    )

    return recommendations


def export_zero1_results_csv(zero1_results, filename):
    """
    Export ZeRO-1 communication analysis results to CSV file.

    Args:
        zero1_results: dict with communication analysis results
        filename: Output CSV filename
    """
    import csv

    with open(filename, 'w', newline='') as f:
        fieldnames = [
            'hardware', 'variant', 'model_mem_gb', 'viable_micros',
            'dp', 'ag_volume_mb', 'ag_time_ms', 'overhead_rating'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for hw_name, hw_data in zero1_results.get("hardware", {}).items():
            for variant_name, variant_data in hw_data.items():
                if variant_data["viable"]:
                    for metrics in variant_data["comm_metrics"]:
                        writer.writerow({
                            'hardware': hw_name,
                            'variant': variant_name,
                            'model_mem_gb': variant_data['model_mem_gb'],
                            'viable_micros': str(variant_data['viable_micros']),
                            'dp': metrics['dp'],
                            'ag_volume_mb': metrics['ag_volume_mb'],
                            'ag_time_ms': metrics['ag_time_ms'],
                            'overhead_rating': metrics['overhead_rating']
                        })
