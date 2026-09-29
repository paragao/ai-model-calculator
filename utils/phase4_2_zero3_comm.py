"""
Phase 4.2: ZeRO-3 Communication Overhead Analysis functions.

Analyzes the communication overhead of ZeRO-3 (Pos+g+p -- optimizer states,
gradients AND parameters sharded) across different data parallelism degrees.

ZeRO-3 is the only stage that pays MORE than baseline data parallelism. The ZeRO
paper (arXiv:1910.02054v3) sec 7.2.2 derives its volume as 3*Psi per step against
2*Psi for ZeRO-1/ZeRO-2 and for baseline DP, i.e. 1.5x -- because the sharded
parameters must be re-gathered twice per step, once for the forward pass and
again for the backward pass, on top of the gradient reduce-scatter that every
stage pays.
"""

# Import configurations
from configuration.project_config import *
from configuration.advanced_config import *

# Import core calculation functions
from .core_calculations import (
    calculate_model_memory_gb,
    calculate_viable_micro_batch_sizes
)

# Import formatting utilities
from .formatting_utils import color_text


def analyze_zero3_communication_overhead(variant, hw, dp_values, layers_per_gpu, experts_per_gpu):
    """
    Analyze ZeRO-3 communication overhead for a variant/hardware combination.

    ZeRO-3 shards optimizer states, gradients and parameters. Communication is a
    parameter all-gather during the forward pass, a second parameter all-gather
    during the backward pass, and a gradient reduce-scatter.

    Args:
        variant: Variant dict with model architecture
        hw: Hardware dict with GPU specs
        dp_values: List of DP values to analyze
        layers_per_gpu: Number of layers per GPU
        experts_per_gpu: Number of experts per GPU

    Returns:
        dict with:
        - viable: bool indicating if variant fits with ZeRO-3
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
        tp=TP,
    )

    # Check viable micro batch sizes with ZeRO-3
    viable_micros = calculate_viable_micro_batch_sizes(
        variant, gpu_mem, dp_full, layers_per_gpu, experts_per_gpu, zero_strategy=3,
        tp=TP
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
        metrics = calculate_zero3_metrics(
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


def calculate_zero3_metrics(model_mem_gb, dp_val, inter_node_bw_gb):
    """
    Calculate ZeRO-3 communication volume and time for one DP degree.

    ZeRO paper (arXiv:1910.02054v3) sec 7.2.2, verbatim:
        "After parameter partitioning, each data parallel process only stores the
         parameters that it updates. Therefore, during the forward propagation it
         needs to receives the parameters for all the other partitions. ...
         The total communication volume is thus (Psi x Nd)/Nd = Psi. ... Note
         however that this all-gather needs to happen once again for the backward
         propagation in the reverse order. The total communication volume is
         therefore the sum of the communication volumes incurred by these
         all-gathers in addition to the communication volume incurred by the
         reduce-scatter of the gradients. The total volume is therefore 3*Psi
         which is 1.5x compared to the baseline."

    So the step decomposes into three collectives of one parameter replica each:
      1. forward-pass parameter all-gather   -- Psi
      2. backward-pass parameter all-gather  -- Psi
      3. gradient reduce-scatter             -- Psi
    Each is reported separately so the caller can see WHERE the extra 1.5x comes
    from: it is the two parameter all-gathers, not a more expensive reduce-scatter
    (item 3 is identical to the single collective ZeRO-2 pays).

    As in phases 4 and 4.1, the paper states Psi while the exact pipelined-ring
    figure is (DP-1)/DP * Psi, used here, which approaches Psi as DP grows. It
    does NOT shrink with DP -- ZeRO-DP volume is independent of GPU count.

    Args:
        model_mem_gb: Model memory per GPU in GB (one parameter replica, Psi)
        dp_val: Data parallelism degree
        inter_node_bw_gb: Inter-node bandwidth in GB/s

    Returns:
        dict with:
        - fwd_ag_volume_mb: Forward parameter all-gather volume in MB per step
        - bwd_ag_volume_mb: Backward parameter all-gather volume in MB per step
        - rs_volume_mb: Gradient reduce-scatter volume in MB per step
        - total_volume_mb: Sum of the three, i.e. 3*Psi
        - total_time_ms: Time for the total volume in milliseconds
        - baseline_ratio: total volume / baseline DP volume (2*Psi), ~1.5
        - overhead_rating: Performance rating (0=excellent, 3=poor)
    """
    ring_fraction = (dp_val - 1) / dp_val if dp_val > 1 else 0.0
    one_replica_mb = model_mem_gb * ring_fraction * 1000

    # The three collectives of sec 7.2.2, one parameter replica each.
    fwd_ag_volume_mb = one_replica_mb
    bwd_ag_volume_mb = one_replica_mb
    rs_volume_mb = one_replica_mb
    total_volume_mb = fwd_ag_volume_mb + bwd_ag_volume_mb + rs_volume_mb

    # Communication time at inter-node bandwidth, derived from the volume above
    # so the two cannot drift apart.
    total_time_ms = total_volume_mb / inter_node_bw_gb if inter_node_bw_gb else 0.0

    # Ratio against baseline DP, which the paper puts at 2*Psi (sec 7.1). This is
    # 1.5 by construction (3/2) and is reported to make the paper's headline
    # "modest 50% increase in communication volume" checkable in the output rather
    # than asserted in prose.
    baseline_volume_mb = 2 * one_replica_mb
    baseline_ratio = (total_volume_mb / baseline_volume_mb) if baseline_volume_mb else 0.0

    # Overhead rating. Thresholds are 3x those of phase 4.1, because this phase
    # reports THREE collectives where phase 4.1 reports one -- so a config rated
    # "excellent" here moves the same per-collective volume per unit time as one
    # rated "excellent" there, and the ratings stay comparable across phases.
    if total_time_ms < 4.5:
        overhead_rating = 0  # Excellent (<4.5ms)
    elif total_time_ms < 9.0:
        overhead_rating = 1  # Good (4.5-9ms)
    elif total_time_ms < 21.0:
        overhead_rating = 2  # Moderate (9-21ms)
    else:
        overhead_rating = 3  # High (>21ms)

    return {
        "fwd_ag_volume_mb": fwd_ag_volume_mb,
        "bwd_ag_volume_mb": bwd_ag_volume_mb,
        "rs_volume_mb": rs_volume_mb,
        "total_volume_mb": total_volume_mb,
        "total_time_ms": total_time_ms,
        "baseline_ratio": baseline_ratio,
        "overhead_rating": overhead_rating
    }


def generate_zero3_recommendations(zero3_results):
    """
    Analyze ZeRO-3 communication results and generate recommendations.

    Args:
        zero3_results: dict with ZeRO-3 communication analysis results

    Returns:
        list of recommendation strings
    """
    recommendations = []

    if not zero3_results.get("hardware"):
        return ["⚠️  No ZeRO-3 communication analysis available"]

    total_configs = 0
    excellent_configs = 0
    high_overhead_configs = 0

    for hw_name, hw_data in zero3_results["hardware"].items():
        for variant_name, variant_data in hw_data.items():
            if variant_data["viable"]:
                for metrics in variant_data["comm_metrics"]:
                    total_configs += 1
                    rating = metrics["overhead_rating"]
                    if rating == 0:
                        excellent_configs += 1
                    elif rating == 3:
                        high_overhead_configs += 1

    if total_configs > 0:
        excellent_pct = (excellent_configs / total_configs) * 100
        high_pct = (high_overhead_configs / total_configs) * 100

        if excellent_pct >= 60:
            recommendations.append(
                f"✅ ZeRO-3 communication overhead is excellent for {excellent_configs}/{total_configs} configs (<4.5ms)"
            )
        elif high_pct >= 30:
            recommendations.append(
                f"⚠️  ZeRO-3 communication overhead is high for {high_overhead_configs}/{total_configs} configs (>21ms)"
            )

    # The central trade-off, stated as the paper derives it rather than as a
    # vague efficiency claim.
    recommendations.append(
        "📐 ZeRO-3 moves 3*Psi per step vs 2*Psi for ZeRO-1/ZeRO-2 and for baseline DP "
        "(ZeRO paper sec 7.2.2) → a fixed 1.5x communication cost, bought with "
        "memory that falls linearly in DP degree"
    )

    # ZeRO-3's extra cost is entirely the two parameter all-gathers.
    recommendations.append(
        "🔍 The extra 1.5x is the two parameter all-gathers (forward + backward), not a "
        "costlier reduce-scatter → the gradient collective is the same size ZeRO-2 pays"
    )

    # Volume does not scale down with DP. Phases 4 and 4.1 used to claim the
    # opposite; it is wrong for every ZeRO stage.
    recommendations.append(
        "⚖️  ZeRO-DP volume per rank is INDEPENDENT of DP degree (ring collectives move "
        "(DP-1)/DP of a replica, approaching 1) → raising DP does not reduce "
        "per-GPU communication, it only reduces memory"
    )

    # Selection guidance: ZeRO-3 is the fallback for models that do not otherwise fit.
    recommendations.append(
        "💡 Prefer ZeRO-1/ZeRO-2 when the model fits: same 2*Psi volume, no parameter "
        "re-gathering. Choose ZeRO-3 when parameter memory is what blocks the fit"
    )

    return recommendations


def export_zero3_results_csv(zero3_results, filename):
    """
    Export ZeRO-3 communication analysis results to CSV file.

    Args:
        zero3_results: dict with communication analysis results
        filename: Output CSV filename
    """
    import csv

    with open(filename, 'w', newline='') as f:
        fieldnames = [
            'hardware', 'variant', 'model_mem_gb', 'viable_micros',
            'dp', 'fwd_ag_volume_mb', 'bwd_ag_volume_mb', 'rs_volume_mb',
            'total_volume_mb', 'total_time_ms', 'baseline_ratio', 'overhead_rating'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for hw_name, hw_data in zero3_results.get("hardware", {}).items():
            for variant_name, variant_data in hw_data.items():
                if variant_data["viable"]:
                    for metrics in variant_data["comm_metrics"]:
                        writer.writerow({
                            'hardware': hw_name,
                            'variant': variant_name,
                            'model_mem_gb': variant_data['model_mem_gb'],
                            'viable_micros': str(variant_data['viable_micros']),
                            'dp': metrics['dp'],
                            'fwd_ag_volume_mb': metrics['fwd_ag_volume_mb'],
                            'bwd_ag_volume_mb': metrics['bwd_ag_volume_mb'],
                            'rs_volume_mb': metrics['rs_volume_mb'],
                            'total_volume_mb': metrics['total_volume_mb'],
                            'total_time_ms': metrics['total_time_ms'],
                            'baseline_ratio': metrics['baseline_ratio'],
                            'overhead_rating': metrics['overhead_rating']
                        })
