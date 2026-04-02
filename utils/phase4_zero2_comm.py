"""
Phase 4: ZeRO-2 Communication Overhead Analysis functions.

Analyzes reduce-scatter communication overhead for ZeRO-2 optimization strategy
across different data parallelism degrees.
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


def analyze_communication_overhead(variant, hw, dp_values, layers_per_gpu, experts_per_gpu):
    """
    Analyze ZeRO-2 communication overhead for a variant/hardware combination.

    Args:
        variant: Variant dict with model architecture
        hw: Hardware dict with GPU specs
        dp_values: List of DP values to analyze
        layers_per_gpu: Number of layers per GPU
        experts_per_gpu: Number of experts per GPU

    Returns:
        dict with:
        - viable: bool indicating if variant fits with ZeRO-2
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

    # Check viable micro batch sizes with ZeRO-2
    viable_micros = calculate_viable_micro_batch_sizes(
        variant, gpu_mem, dp_full, layers_per_gpu, experts_per_gpu, zero_strategy=2
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
        metrics = calculate_communication_metrics(
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


def calculate_communication_metrics(model_mem_gb, dp_val, inter_node_bw_gb):
    """
    Calculate ZeRO-2 reduce-scatter communication volume and time.

    Args:
        model_mem_gb: Model memory per GPU in GB
        dp_val: Data parallelism degree
        inter_node_bw_gb: Inter-node bandwidth in GB/s

    Returns:
        dict with:
        - rs_volume_mb: Reduce-scatter volume in MB per micro-batch
        - rs_time_ms: Reduce-scatter time in milliseconds
        - overhead_rating: Performance rating (0=excellent, 3=poor)
    """
    # Reduce-scatter volume: model + gradients = 2x model memory / DP
    rs_volume_mb = model_mem_gb * 2 / dp_val * 1000

    # Communication time at inter-node bandwidth
    rs_time_ms = model_mem_gb * 2 / dp_val / inter_node_bw_gb * 1000

    # Assign overhead rating based on communication time
    # These thresholds are heuristics based on typical compute times
    if rs_time_ms < 2.0:
        overhead_rating = 0  # Excellent (<2ms)
    elif rs_time_ms < 5.0:
        overhead_rating = 1  # Good (2-5ms)
    elif rs_time_ms < 10.0:
        overhead_rating = 2  # Moderate (5-10ms)
    else:
        overhead_rating = 3  # High (>10ms)

    return {
        "rs_volume_mb": rs_volume_mb,
        "rs_time_ms": rs_time_ms,
        "overhead_rating": overhead_rating
    }


def generate_communication_recommendations(comm_results):
    """
    Analyze ZeRO-2 communication results and generate recommendations.

    Args:
        comm_results: dict with variant/hardware communication analysis

    Returns:
        list of recommendation strings
    """
    recommendations = []

    if not comm_results.get("hardware"):
        return ["⚠️  No communication analysis available"]

    # Analyze overall communication patterns
    total_configs = 0
    excellent_configs = 0
    good_configs = 0
    moderate_configs = 0
    high_overhead_configs = 0

    for hw_name, hw_data in comm_results["hardware"].items():
        for variant_name, variant_data in hw_data.items():
            if variant_data["viable"]:
                for metrics in variant_data["comm_metrics"]:
                    total_configs += 1
                    rating = metrics["overhead_rating"]
                    if rating == 0:
                        excellent_configs += 1
                    elif rating == 1:
                        good_configs += 1
                    elif rating == 2:
                        moderate_configs += 1
                    else:
                        high_overhead_configs += 1

    if total_configs > 0:
        excellent_pct = (excellent_configs / total_configs) * 100
        high_pct = (high_overhead_configs / total_configs) * 100

        if excellent_pct >= 50:
            recommendations.append(
                f"✅ ZeRO-2 communication overhead is excellent for {excellent_configs}/{total_configs} configs (<2ms)"
            )
        elif high_pct >= 30:
            recommendations.append(
                f"⚠️  ZeRO-2 communication overhead is high for {high_overhead_configs}/{total_configs} configs (>10ms)"
            )

    # Find optimal DP configurations
    best_dp_by_hw = {}
    for hw_name, hw_data in comm_results["hardware"].items():
        best_time = float('inf')
        best_dp = None

        for variant_name, variant_data in hw_data.items():
            if variant_data["viable"]:
                for metrics in variant_data["comm_metrics"]:
                    if metrics["rs_time_ms"] < best_time:
                        best_time = metrics["rs_time_ms"]
                        best_dp = metrics["dp"]

        if best_dp:
            best_dp_by_hw[hw_name] = (best_dp, best_time)

    if best_dp_by_hw:
        recommendations.append(
            f"🎯 Optimal DP configurations minimize communication: "
            f"Higher DP = lower per-GPU communication volume"
        )

    # Identify variants with problematic communication
    problematic_variants = []
    for hw_name, hw_data in comm_results["hardware"].items():
        for variant_name, variant_data in hw_data.items():
            if variant_data["viable"]:
                # Check if any DP value has high overhead
                has_high_overhead = any(
                    m["overhead_rating"] >= 3 for m in variant_data["comm_metrics"]
                )
                if has_high_overhead:
                    if variant_name not in problematic_variants:
                        problematic_variants.append(variant_name)

    if problematic_variants:
        recommendations.append(
            f"📊 Variants with high communication overhead at low DP: {', '.join(problematic_variants)} "
            f"→ Use higher DP or consider ZeRO-1 if memory permits"
        )

    return recommendations


def export_communication_results_csv(comm_results, filename):
    """
    Export ZeRO-2 communication analysis results to CSV file.

    Args:
        comm_results: dict with communication analysis results
        filename: Output CSV filename
    """
    import csv

    with open(filename, 'w', newline='') as f:
        fieldnames = [
            'hardware', 'variant', 'model_mem_gb', 'viable_micros',
            'dp', 'rs_volume_mb', 'rs_time_ms', 'overhead_rating'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for hw_name, hw_data in comm_results.get("hardware", {}).items():
            for variant_name, variant_data in hw_data.items():
                if variant_data["viable"]:
                    for metrics in variant_data["comm_metrics"]:
                        writer.writerow({
                            'hardware': hw_name,
                            'variant': variant_name,
                            'model_mem_gb': variant_data['model_mem_gb'],
                            'viable_micros': str(variant_data['viable_micros']),
                            'dp': metrics['dp'],
                            'rs_volume_mb': metrics['rs_volume_mb'],
                            'rs_time_ms': metrics['rs_time_ms'],
                            'overhead_rating': metrics['overhead_rating']
                        })
