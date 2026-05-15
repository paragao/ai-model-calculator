"""
Phase 1: Memory Analysis functions.

Analyzes memory requirements for model variants across different hardware configurations,
determining optimal ZeRO strategies and micro batch sizes.
"""

# Import configurations
from configuration.project_config import *
from configuration.advanced_config import *

# Import core calculation functions
from .core_calculations import (
    calculate_model_memory_gb,
    calculate_memory_for_micro,
    calculate_viable_micro_batch_sizes
)

# Import formatting utilities
from .formatting_utils import color_text


def calculate_efficiency_metrics(memory_breakdown, gpu_mem):
    """
    Calculate memory efficiency metrics.

    Args:
        memory_breakdown: Dict with model/grad/optim/activation/buffer memory in GB
        gpu_mem: GPU memory capacity in GB

    Returns:
        dict with:
        - memory_efficiency: (model+grad+optim)/total (computational memory vs total)
        - utilization: total/gpu_capacity (percentage of GPU used)
        - utilization_pct: utilization as percentage
        - wasted_headroom: gpu_capacity - total (unused memory)
    """
    total = memory_breakdown["total"]
    computational_memory = (
        memory_breakdown["model"]
        + memory_breakdown["grad"]
        + memory_breakdown["optim"]
    )

    utilization = total / gpu_mem if gpu_mem > 0 else 0
    memory_efficiency = computational_memory / total if total > 0 else 0

    return {
        "memory_efficiency": memory_efficiency,
        "utilization": utilization,
        "utilization_pct": utilization * 100,
        "wasted_headroom": gpu_mem - total,
    }


def diagnose_oom(variant, hw, dp, layers_per_gpu, experts_per_gpu,
                 attempted_zero_strategies=[1, 2]):
    """
    Generate detailed OOM diagnostics and suggestions.

    Args:
        variant: Variant dictionary
        hw: Hardware dictionary
        dp: Data parallelism degree
        layers_per_gpu: Layers per GPU
        experts_per_gpu: Experts per GPU
        attempted_zero_strategies: Which ZeRO strategies were tried

    Returns:
        dict with:
        - available_memory: GPU capacity
        - required_memory: Memory needed for smallest config
        - shortage: Gap to overcome
        - breakdown: Component-wise memory requirements
        - suggestions: List of actionable fixes
    """
    gpu_mem = hw["mem_gb"]

    # Calculate model memory (doesn't change with ZeRO)
    model_mem_gb = calculate_model_memory_gb(
        variant, layers_per_gpu, experts_per_gpu,
        PARAM_BYTES, VOCAB, LAYER_NORM, FFN_WEIGHT_MATRICES, NUM_EMBEDDINGS_TABLES,
        tp=TP
    )

    # Try to calculate with micro=1 and best ZeRO strategy attempted
    best_zero = max(attempted_zero_strategies) if attempted_zero_strategies else 2

    memory_breakdown = calculate_memory_for_micro(
        micro=1,
        model_mem_gb=model_mem_gb,
        variant=variant,
        dp=dp,
        layers_per_gpu=layers_per_gpu,
        zero_strategy=best_zero,
        seq_len=SEQ_LEN,
        topk=TOPK,
        param_bytes=PARAM_BYTES,
        empirical_act_multiplier=EMPIRICAL_ACT_MULTIPLIER,
        selective_act_checkpointing_multiplier=SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER,
        fwd_bwd_routing_buff_passes=FWD_BWD_ROUTING_BUFF_PASSES,
        nccl_mem_buf=NCCL_MEM_BUF,
        optim_bytes=OPTIM_BYTES,
    )

    required = memory_breakdown["total"]
    shortage = required - gpu_mem

    # Generate suggestions
    suggestions = []

    # Suggestion 1: Try ZeRO-3 if not already tried
    if 3 not in attempted_zero_strategies:
        # ZeRO-3 shards model parameters too
        potential_savings = model_mem_gb * (1 - 1/dp) if dp > 1 else model_mem_gb * 0.5
        suggestions.append(
            f"Use ZeRO-3 to shard model parameters (could save ~{potential_savings:.1f}G)"
        )

    # Suggestion 2: Reduce layers
    if shortage > 0:
        reduction_pct = min(30, int((shortage / required) * 100) + 10)
        suggestions.append(
            f"Reduce model layers by {reduction_pct}% (saves ~{shortage * 0.5:.1f}G)"
        )

    # Suggestion 3: Reduce hidden dimension
    suggestions.append(
        f"Reduce hidden dimension (d={variant['d']}) by 10% (saves ~{required * 0.19:.1f}G)"
    )

    # Suggestion 4: Switch to better hardware
    if hw["mem_gb"] < 100:
        suggestions.append(
            f"Switch to H200 ({H200_MEM_GB}GB vs {hw['mem_gb']}GB)"
        )

    # Suggestion 5: Reduce micro batch or use gradient checkpointing
    act_mem = memory_breakdown["activation"]
    if act_mem > 10:
        suggestions.append(
            f"Use more aggressive activation checkpointing (could save ~{act_mem * 0.3:.1f}G)"
        )

    return {
        "available_memory": gpu_mem,
        "required_memory": required,
        "shortage": shortage,
        "breakdown": memory_breakdown,
        "suggestions": suggestions,
    }


def analyze_variant_memory(variant, hw, dp, layers_per_gpu, experts_per_gpu):
    """
    Analyze memory requirements for a single variant/hardware combination.

    Args:
        variant: Variant dictionary with architecture specs
        hw: Hardware dictionary with GPU specs
        dp: Data parallelism degree
        layers_per_gpu: Number of layers per GPU (total_layers // PP)
        experts_per_gpu: Number of experts per GPU (N_EXPERTS // EP)

    Returns:
        dict with:
        - viable_micros_z1: List of viable micro batch sizes for ZeRO-1
        - viable_micros_z2: List of viable micro batch sizes for ZeRO-2
        - best_zero: Optimal ZeRO strategy (1, 2, or -1 for OOM)
        - best_micro: Largest viable micro batch size
        - memory_breakdown: Dict with model/grad/optim/activation/buffer memory
        - efficiency_metrics: Dict with utilization/efficiency stats
        - oom_diagnostics: Dict with OOM reason and suggestions (if OOM)
        - model_mem_gb: Model parameter memory (for reference)
    """
    gpu_mem = hw["mem_gb"]

    # Calculate model memory once
    model_mem_gb = calculate_model_memory_gb(
        variant, layers_per_gpu, experts_per_gpu,
        PARAM_BYTES, VOCAB, LAYER_NORM, FFN_WEIGHT_MATRICES, NUM_EMBEDDINGS_TABLES,
        tp=TP
    )

    # Try ZeRO-1 first (better efficiency)
    viable_z1 = calculate_viable_micro_batch_sizes(
        variant, gpu_mem, dp, layers_per_gpu, experts_per_gpu, zero_strategy=1,
        tp=TP
    )

    # Try ZeRO-2
    viable_z2 = calculate_viable_micro_batch_sizes(
        variant, gpu_mem, dp, layers_per_gpu, experts_per_gpu, zero_strategy=2,
        tp=TP
    )

    # Determine best ZeRO strategy and micro batch size
    if viable_z1:
        best_zero = 1
        best_micro = max(viable_z1)
    elif viable_z2:
        best_zero = 2
        best_micro = max(viable_z2)
    else:
        # OOM - generate diagnostics
        best_zero = -1
        best_micro = None

        oom_diagnostics = diagnose_oom(
            variant, hw, dp, layers_per_gpu, experts_per_gpu,
            attempted_zero_strategies=[1, 2]
        )

        return {
            "viable_micros_z1": viable_z1,
            "viable_micros_z2": viable_z2,
            "best_zero": best_zero,
            "best_micro": best_micro,
            "memory_breakdown": None,
            "efficiency_metrics": None,
            "oom_diagnostics": oom_diagnostics,
            "model_mem_gb": model_mem_gb,
        }

    # Calculate detailed memory breakdown for best configuration
    memory_breakdown = calculate_memory_for_micro(
        micro=best_micro,
        model_mem_gb=model_mem_gb,
        variant=variant,
        dp=dp,
        layers_per_gpu=layers_per_gpu,
        zero_strategy=best_zero,
        seq_len=SEQ_LEN,
        topk=TOPK,
        param_bytes=PARAM_BYTES,
        empirical_act_multiplier=EMPIRICAL_ACT_MULTIPLIER,
        selective_act_checkpointing_multiplier=SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER,
        fwd_bwd_routing_buff_passes=FWD_BWD_ROUTING_BUFF_PASSES,
        nccl_mem_buf=NCCL_MEM_BUF,
        optim_bytes=OPTIM_BYTES,
    )

    # Calculate efficiency metrics
    efficiency_metrics = calculate_efficiency_metrics(memory_breakdown, gpu_mem)

    return {
        "viable_micros_z1": viable_z1,
        "viable_micros_z2": viable_z2,
        "best_zero": best_zero,
        "best_micro": best_micro,
        "memory_breakdown": memory_breakdown,
        "efficiency_metrics": efficiency_metrics,
        "oom_diagnostics": None,
        "model_mem_gb": model_mem_gb,
    }


def generate_recommendations(all_results):
    """
    Analyze all results and generate actionable recommendations.

    Args:
        all_results: Dict with hardware and variant results

    Returns:
        List of recommendation strings
    """
    recommendations = []

    # Analyze each variant across all hardware
    for variant_name, results_list in all_results["variants"].items():
        # Find best configuration for this variant
        best_config = None
        best_headroom = -1
        oom_count = 0

        for result in results_list:
            if result["best_zero"] == -1:
                oom_count += 1
            elif result["efficiency_metrics"]:
                headroom = result["efficiency_metrics"]["wasted_headroom"]
                if headroom > best_headroom:
                    best_headroom = headroom
                    best_config = result

        # Generate recommendations based on findings
        if oom_count == len(results_list):
            # All configs OOM
            recommendations.append(
                f"⚠️  Variant {variant_name}: OOM on all hardware - consider reducing model size"
            )
        elif oom_count > 0:
            # Some configs OOM
            successful = len(results_list) - oom_count
            recommendations.append(
                f"⚠️  Variant {variant_name}: Only fits on {successful}/{len(results_list)} hardware configs"
            )
        elif best_config and best_headroom > 25:
            # Excessive headroom
            recommendations.append(
                f"💡 Variant {variant_name}: Large headroom ({best_headroom:.1f}G) - could increase micro batch or model size"
            )

        # Check if using suboptimal ZeRO strategy
        if best_config and best_config["best_zero"] == 2 and best_config["viable_micros_z1"]:
            recommendations.append(
                f"💡 Variant {variant_name}: Using ZeRO-2 but ZeRO-1 available - consider hardware with more memory for 10% efficiency gain"
            )

        # Check throughput
        if best_config and best_config["best_micro"] == 1:
            recommendations.append(
                f"⚡ Variant {variant_name}: micro=1 limits throughput - consider hardware with more memory"
            )

    return recommendations


def export_results_csv(results, filename):
    """
    Export analysis results to CSV.

    Args:
        results: Dict with all analysis results
        filename: Output CSV filename
    """
    import csv

    with open(filename, 'w', newline='') as f:
        fieldnames = [
            'variant', 'hardware', 'zero_strategy', 'micro_batch',
            'model_gb', 'grad_gb', 'optim_gb', 'activation_gb', 'buffer_gb',
            'total_gb', 'gpu_capacity_gb', 'headroom_gb', 'utilization_pct',
            'memory_efficiency', 'status'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for variant_name, results_list in results["variants"].items():
            for idx, result in enumerate(results_list):
                hw_name = results["hardware"][idx] if idx < len(results["hardware"]) else "Unknown"

                if result["best_zero"] == -1:
                    # OOM case
                    row = {
                        'variant': variant_name,
                        'hardware': hw_name,
                        'zero_strategy': 'OOM',
                        'micro_batch': 'N/A',
                        'model_gb': result["model_mem_gb"],
                        'grad_gb': 0,
                        'optim_gb': 0,
                        'activation_gb': 0,
                        'buffer_gb': 0,
                        'total_gb': result["oom_diagnostics"]["required_memory"],
                        'gpu_capacity_gb': result["oom_diagnostics"]["available_memory"],
                        'headroom_gb': -result["oom_diagnostics"]["shortage"],
                        'utilization_pct': 0,
                        'memory_efficiency': 0,
                        'status': 'OOM'
                    }
                else:
                    # Success case
                    mb = result["memory_breakdown"]
                    em = result["efficiency_metrics"]
                    row = {
                        'variant': variant_name,
                        'hardware': hw_name,
                        'zero_strategy': f'ZeRO-{result["best_zero"]}',
                        'micro_batch': result["best_micro"],
                        'model_gb': mb["model"],
                        'grad_gb': mb["grad"],
                        'optim_gb': mb["optim"],
                        'activation_gb': mb["activation"],
                        'buffer_gb': mb["buffer"],
                        'total_gb': mb["total"],
                        'gpu_capacity_gb': mb["total"] + em["wasted_headroom"],
                        'headroom_gb': em["wasted_headroom"],
                        'utilization_pct': em["utilization_pct"],
                        'memory_efficiency': em["memory_efficiency"],
                        'status': 'OK'
                    }

                writer.writerow(row)

    print(f"\n✅ Exported results to {filename}")


def export_results_json(results, filename):
    """
    Export full results structure to JSON.

    Args:
        results: Dict with all analysis results
        filename: Output JSON filename
    """
    import json

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Exported results to {filename}")
