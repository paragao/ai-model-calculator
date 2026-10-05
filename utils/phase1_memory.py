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
    calculate_viable_micro_batch_sizes,
    calculate_zero3_transient_gb,
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
        PARAM_STORAGE_BYTES, VOCAB, LAYER_NORM, FFN_WEIGHT_MATRICES, NUM_EMBEDDINGS_TABLES,
        tp=TP,
        pp=PP,
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
        seq_len=variant.get("seq_len", SEQ_LEN),
        # FALLBACK. calculate_memory_for_micro resolves the variant's own
        # "topk" over this via resolve_topk, exactly as it does the expert
        # count, so the global applies only to a variant declaring none.
        topk=TOPK,
        param_bytes=PARAM_STORAGE_BYTES,
        empirical_act_multiplier=EMPIRICAL_ACT_MULTIPLIER,
        selective_act_checkpointing_multiplier=SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER,
        fwd_bwd_routing_buff_passes=FWD_BWD_ROUTING_BUFF_PASSES,
        nccl_mem_buf=NCCL_MEM_BUF,
        optim_bytes=OPTIM_BYTES,
        ep=EP,
        pp=PP,
        vp=VP,
        nccl_ep_scaling_factor=NCCL_EP_SCALING_FACTOR,
        fragmentation_factor=FRAGMENTATION_FACTOR,
        experts_per_gpu=experts_per_gpu,
        grad_bytes=GRAD_BYTES,
        vocab=VOCAB,
        logits_bytes_per_element=LOGITS_BYTES_PER_ELEMENT,
        logits_fused_ce=LOGITS_FUSED_CE,
        logits_chunk_tokens=LOGITS_CHUNK_TOKENS,
        cp=CP,
        act_bytes_per_element=ACT_BYTES_PER_ELEMENT,
        recompute_granularity=RECOMPUTE_GRANULARITY,
        recompute_num_layers=RECOMPUTE_NUM_LAYERS,
        recompute_boundary_bytes_per_element=RECOMPUTE_BOUNDARY_BYTES_PER_ELEMENT,
        act_precision_saving_fraction=ACT_PRECISION_SAVING_FRACTION,
    )

    required = memory_breakdown["total"]
    shortage = required - gpu_mem

    # Generate suggestions
    suggestions = []

    # Suggestion 1: Try ZeRO-3 if not already tried
    #
    # NOTE -- the only in-tree caller (analyze_variant_memory) passes
    # attempted_zero_strategies=[1, 2, 3], so this branch is currently unreachable
    # from the calculator itself and fires only for an external caller using the
    # [1, 2] default. Flagged rather than removed; the arithmetic below is still
    # corrected because wrong advice is worse than unreachable advice.
    if 3 not in attempted_zero_strategies:
        # ZeRO-3 shards resident parameters to Psi/Nd, but adds back the transient
        # all-gather buffer, which does NOT shrink with DP. Quote the NET saving:
        # at large DP with a small model the buffer can exceed the resident saving,
        # in which case ZeRO-3 costs memory rather than saving it.
        resident_saving = model_mem_gb * (1 - 1/dp) if dp > 1 else model_mem_gb * 0.5
        transient = calculate_zero3_transient_gb(
            variant=variant,
            experts_per_gpu=experts_per_gpu,
            param_bytes=PARAM_STORAGE_BYTES,
            vocab=VOCAB,
            layer_norm=LAYER_NORM,
            ffn_weight_matrices=FFN_WEIGHT_MATRICES,
            max_live_params=ZERO3_MAX_LIVE_PARAMS,
            prefetch_bucket_params=ZERO3_PREFETCH_BUCKET_PARAMS,
            tp=TP,
        )["transient_gb"]
        net_saving = resident_saving - transient
        if net_saving > 0:
            suggestions.append(
                f"Use ZeRO-3 to shard model parameters (could save ~{net_saving:.1f}G "
                f"net: {resident_saving:.1f}G resident less {transient:.1f}G "
                f"transient all-gather buffer)"
            )
        else:
            suggestions.append(
                f"ZeRO-3 would NOT help here: its {transient:.1f}G transient "
                f"all-gather buffer exceeds the {resident_saving:.1f}G it saves on "
                f"resident parameters at DP={dp}"
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
        - viable_micros_z3: List of viable micro batch sizes for ZeRO-3
        - best_zero: Optimal ZeRO strategy (1, 2, 3, or -1 for OOM). Strategies
            are tried in ascending order, so the lowest-communication strategy
            that fits wins; 3 means only parameter sharding made it fit.
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
        PARAM_STORAGE_BYTES, VOCAB, LAYER_NORM, FFN_WEIGHT_MATRICES, NUM_EMBEDDINGS_TABLES,
        tp=TP,
        pp=PP,
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

    # Try ZeRO-3, which additionally shards the resident parameters. Only reached
    # when ZeRO-1 and ZeRO-2 both OOM: it fits strictly more than they do, but
    # costs 3*Psi per step against their 2*Psi (see phase 4.2), so it is a last
    # resort rather than a default.
    viable_z3 = calculate_viable_micro_batch_sizes(
        variant, gpu_mem, dp, layers_per_gpu, experts_per_gpu, zero_strategy=3,
        tp=TP
    )

    # Determine best ZeRO strategy and micro batch size
    if viable_z1:
        best_zero = 1
        best_micro = max(viable_z1)
    elif viable_z2:
        best_zero = 2
        best_micro = max(viable_z2)
    elif viable_z3:
        best_zero = 3
        best_micro = max(viable_z3)
    else:
        # OOM - generate diagnostics
        best_zero = -1
        best_micro = None

        oom_diagnostics = diagnose_oom(
            variant, hw, dp, layers_per_gpu, experts_per_gpu,
            attempted_zero_strategies=[1, 2, 3]
        )

        return {
            "viable_micros_z1": viable_z1,
            "viable_micros_z2": viable_z2,
            "viable_micros_z3": viable_z3,
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
        seq_len=variant.get("seq_len", SEQ_LEN),
        # FALLBACK. calculate_memory_for_micro resolves the variant's own
        # "topk" over this via resolve_topk, exactly as it does the expert
        # count, so the global applies only to a variant declaring none.
        topk=TOPK,
        param_bytes=PARAM_STORAGE_BYTES,
        empirical_act_multiplier=EMPIRICAL_ACT_MULTIPLIER,
        selective_act_checkpointing_multiplier=SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER,
        fwd_bwd_routing_buff_passes=FWD_BWD_ROUTING_BUFF_PASSES,
        nccl_mem_buf=NCCL_MEM_BUF,
        optim_bytes=OPTIM_BYTES,
        ep=EP,
        pp=PP,
        vp=VP,
        nccl_ep_scaling_factor=NCCL_EP_SCALING_FACTOR,
        fragmentation_factor=FRAGMENTATION_FACTOR,
        experts_per_gpu=experts_per_gpu,
        grad_bytes=GRAD_BYTES,
        vocab=VOCAB,
        logits_bytes_per_element=LOGITS_BYTES_PER_ELEMENT,
        logits_fused_ce=LOGITS_FUSED_CE,
        logits_chunk_tokens=LOGITS_CHUNK_TOKENS,
        cp=CP,
        act_bytes_per_element=ACT_BYTES_PER_ELEMENT,
        recompute_granularity=RECOMPUTE_GRANULARITY,
        recompute_num_layers=RECOMPUTE_NUM_LAYERS,
        recompute_boundary_bytes_per_element=RECOMPUTE_BOUNDARY_BYTES_PER_ELEMENT,
        act_precision_saving_fraction=ACT_PRECISION_SAVING_FRACTION,
    )

    # Calculate efficiency metrics
    efficiency_metrics = calculate_efficiency_metrics(memory_breakdown, gpu_mem)

    return {
        "viable_micros_z1": viable_z1,
        "viable_micros_z2": viable_z2,
        "viable_micros_z3": viable_z3,
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
        best_headroom = float("inf")
        oom_count = 0

        for result in results_list:
            if result["best_zero"] == -1:
                oom_count += 1
            elif result["efficiency_metrics"]:
                headroom = result["efficiency_metrics"]["wasted_headroom"]
                # TIGHTEST FIT, not loosest. This read `headroom > best_headroom`,
                # selecting the config with the MOST unused memory and calling it
                # best_config, while model_calculations.py:130 uses `<` for the
                # same concept. The recommendation below then said "micro=1 limits
                # throughput -- consider hardware with more memory" on the basis
                # of whichever preset had the most memory left over: for
                # Qwen2.5-72B that was `1 p6e-gb200` with 55.6 GB wasted, about
                # the largest-memory preset in the catalog. Exactly backwards.
                if headroom < best_headroom:
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
        #
        # NOTE -- this condition is UNREACHABLE: strategies are tried in ascending
        # order, so a non-empty viable_micros_z1 forces best_zero to 1 and it can
        # never be 2 here. Left in place rather than deleted because removing it
        # changes no output; flagged so it is not mistaken for live logic.
        if best_config and best_config["best_zero"] == 2 and best_config["viable_micros_z1"]:
            recommendations.append(
                f"💡 Variant {variant_name}: Using ZeRO-2 but ZeRO-1 available - consider hardware with more memory for 10% efficiency gain"
            )

        # ZeRO-3 fits strictly more than ZeRO-1/ZeRO-2 but costs 3*Psi per step
        # against their 2*Psi (ZeRO paper sec 7.2.2, modelled in phase 4.2), so
        # falling back to it is worth surfacing rather than passing silently.
        if best_config and best_config["best_zero"] == 3:
            recommendations.append(
                f"⚠️  Variant {variant_name}: Fits only with ZeRO-3 (parameter sharding) - "
                f"expect 1.5x the communication volume of ZeRO-1/ZeRO-2; see phase 4.2"
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
            'logits_gb', 'zero3_transient_gb', 'recompute_fraction',
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
                        'logits_gb': 0,
                        'recompute_fraction': '',
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
                        'logits_gb': mb["logits"],
                        'zero3_transient_gb': mb["zero3_transient"],
                        'recompute_fraction': mb["recompute_fraction"],
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
