#!/usr/bin/python3
# ============================================================================
# PHASE 1: Memory analysis scenario
# ============================================================================

# Import configurations
from variants_config import VARIANTS
from hardware_config import HARDWARE
from project_config import *
from advanced_config import *


def calculate_model_memory_gb(
    variant,
    layers_per_gpu,
    experts_per_gpu,
    param_bytes,
    vocab,
    layer_norm,
    ffn_weight_matrices,
    num_embeddings_tables,
):
    """
    Calculate model memory in GB for a given variant.

    Args:
        variant: Variant dictionary with architecture specs
        layers_per_gpu: Number of layers per GPU (total_layers // PP)
        experts_per_gpu: Number of experts per GPU (N_EXPERTS // EP)
        param_bytes: Bytes per parameter (2 for BF16, 1 for FP8)
        vocab: Vocabulary size
        layer_norm: Layer norm constant (typically 2)
        ffn_weight_matrices: FFN weight matrices constant (typically 3)
        num_embeddings_tables: Number of embedding tables (typically 2)

    Returns:
        float: Model memory in GB
    """
    d = variant["d"]
    layers = variant["layers"]
    moe_layers = layers - variant["dense_layers"]

    attn_mem = layers_per_gpu * variant["attn_per_layer"] * param_bytes
    routed_mem = experts_per_gpu * moe_layers * variant["expert_each"] * param_bytes
    shared_mem = moe_layers * variant["shared_each"] * param_bytes
    router_mem = moe_layers * variant["router_each"] * param_bytes
    ln_mem = layers * layer_norm * d * param_bytes
    dense_ffn_mem = (
        variant["dense_layers"]
        * ffn_weight_matrices
        * d
        * variant["dense_ffn"]
        * param_bytes
    )
    embed_mem = num_embeddings_tables * vocab * d * param_bytes

    model_mem_bytes = (
        attn_mem
        + routed_mem
        + shared_mem
        + router_mem
        + ln_mem
        + dense_ffn_mem
        + embed_mem
    )
    return model_mem_bytes / 1e9


def calculate_training_time_months(
    total_tokens, active_params_b, num_gpus, peak_tflops_per_gpu, mfu, efficiency
):
    """
    Calculate training time in months based on FLOPs.

    Args:
        total_tokens: Total tokens to train on (e.g., 30e12 for 30T)
        active_params_b: Active parameters in billions (for MoE, this is params per forward pass)
        num_gpus: Number of GPUs
        peak_tflops_per_gpu: Peak TFLOPs per GPU for the given precision
        mfu: Model FLOPs Utilization (fraction of peak, typically 0.35-0.50)
        efficiency: Training efficiency multiplier (accounts for communication overhead, ZeRO strategy, etc.)

    Returns:
        float: Training time in months
    """
    # Total FLOPs = 6 * tokens * parameters
    # Factor of 6: 1 forward + 2 backward (gradients) + 3 for optimizer updates/recomputation
    total_flops = (
        6 * total_tokens * active_params_b * 1e9
    )  # Convert billions to actual count

    # Achieved TFLOPs/s across all GPUs
    achieved_tflops_per_sec = peak_tflops_per_gpu * num_gpus * mfu * efficiency

    # Training time in seconds
    training_time_seconds = total_flops / (achieved_tflops_per_sec * 1e12)

    # Convert to months (30.44 days per month average)
    training_time_days = training_time_seconds / (24 * 3600)
    training_time_months = training_time_days / 30.44

    return training_time_months


def calculate_memory_for_micro(
    micro,
    model_mem_gb,
    variant,
    dp,
    layers_per_gpu,
    zero_strategy,
    seq_len,
    topk,
    param_bytes,
    empirical_act_multiplier,
    selective_act_checkpointing_multiplier,
    fwd_bwd_routing_buff_passes,
    nccl_mem_buf,
    optim_bytes,
):
    """
    Calculate total GPU memory required for a given micro batch size.

    Args:
        micro: Micro batch size (sequences per GPU)
        model_mem_gb: Model parameter memory in GB
        variant: Variant dictionary with architecture specs
        dp: Data parallelism degree
        layers_per_gpu: Number of layers per GPU (total_layers // PP)
        zero_strategy: ZeRO strategy (1, 2, or 3)
        seq_len: Sequence length
        topk: Top-K for MoE routing
        param_bytes: Bytes per parameter
        empirical_act_multiplier: Activation multiplier constant
        selective_act_checkpointing_multiplier: Checkpointing multiplier
        fwd_bwd_routing_buff_passes: Forward/backward routing buffer passes
        nccl_mem_buf: NCCL buffer memory in GB
        optim_bytes: Optimizer bytes per parameter

    Returns:
        dict: Memory breakdown with keys:
            - total: Total memory in GB
            - model: Model parameter memory
            - grad: Gradient memory
            - optim: Optimizer state memory
            - activation: Activation memory
            - buffer: Communication buffer memory
    """
    d = variant["d"]

    # Calculate gradient and optimizer memory based on ZeRO strategy
    if zero_strategy == 1:
        grad_mem = model_mem_gb  # Full gradients on each GPU
        optim_mem = model_mem_gb * optim_bytes / dp  # Optimizer sharded across DP
    elif zero_strategy == 2:
        grad_mem = model_mem_gb / dp  # Gradients sharded across DP
        optim_mem = model_mem_gb * optim_bytes / dp  # Optimizer sharded across DP
    elif zero_strategy == 3:
        grad_mem = model_mem_gb / dp  # Gradients sharded
        optim_mem = model_mem_gb * optim_bytes / dp  # Optimizer sharded
        # ZeRO-3 also shards model params, but we don't model that here
    else:
        raise ValueError(f"Unknown ZeRO strategy: {zero_strategy}")

    # Calculate activation memory (scales with micro batch size)
    act_per_layer_full = seq_len * micro * d * empirical_act_multiplier
    act_per_layer_selective = (
        act_per_layer_full * selective_act_checkpointing_multiplier
    )
    activation_mem = layers_per_gpu * act_per_layer_selective / 1e9

    # Calculate buffer memory (scales with micro batch size)
    a2a_buf = (
        seq_len * micro * topk * d * fwd_bwd_routing_buff_passes * param_bytes / 1e9
    )
    buffer_mem = a2a_buf + nccl_mem_buf

    # Total memory
    total_mem = model_mem_gb + grad_mem + optim_mem + activation_mem + buffer_mem

    return {
        "total": total_mem,
        "model": model_mem_gb,
        "grad": grad_mem,
        "optim": optim_mem,
        "activation": activation_mem,
        "buffer": buffer_mem,
    }


def calculate_viable_micro_batch_sizes(
    variant,
    gpu_mem_gb,
    dp,
    layers_per_gpu,
    experts_per_gpu,
    zero_strategy,
    micros_to_test=None,
    threshold=None,
):
    """
    Calculate which micro batch sizes fit in GPU memory.

    Args:
        variant: Variant dictionary with architecture specs
        gpu_mem_gb: GPU memory in GB
        dp: Data parallelism degree
        layers_per_gpu: Number of layers per GPU (total_layers // PP)
        experts_per_gpu: Number of experts per GPU (N_EXPERTS // EP)
        zero_strategy: ZeRO strategy (1, 2, or 3)
        micros_to_test: List of micro batch sizes to test (default: use global MICROS)
        threshold: Memory utilization threshold (default: use global GPU_MEM_UTILIZATION_THRESHOLD)

    Returns:
        list: Viable micro batch sizes sorted ascending (e.g., [1, 2, 4])
              Empty list if none fit (OOM scenario)
    """
    if micros_to_test is None:
        micros_to_test = MICROS

    if threshold is None:
        threshold = GPU_MEM_UTILIZATION_THRESHOLD

    # Calculate model memory once (doesn't depend on micro batch size)
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

    viable_micros = []
    for micro in micros_to_test:
        memory_breakdown = calculate_memory_for_micro(
            micro=micro,
            model_mem_gb=model_mem_gb,
            variant=variant,
            dp=dp,
            layers_per_gpu=layers_per_gpu,
            zero_strategy=zero_strategy,
            seq_len=SEQ_LEN,
            topk=TOPK,
            param_bytes=PARAM_BYTES,
            empirical_act_multiplier=EMPIRICAL_ACT_MULTIPLIER,
            selective_act_checkpointing_multiplier=SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER,
            fwd_bwd_routing_buff_passes=FWD_BWD_ROUTING_BUFF_PASSES,
            nccl_mem_buf=NCCL_MEM_BUF,
            optim_bytes=OPTIM_BYTES,
        )

        # Check if this micro batch size fits in memory
        if memory_breakdown["total"] <= gpu_mem_gb * threshold:
            viable_micros.append(micro)

    return sorted(viable_micros)


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
        PARAM_BYTES, VOCAB, LAYER_NORM, FFN_WEIGHT_MATRICES, NUM_EMBEDDINGS_TABLES
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
            f"Switch to H200 (141GB vs {hw['mem_gb']}GB)"
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
        PARAM_BYTES, VOCAB, LAYER_NORM, FFN_WEIGHT_MATRICES, NUM_EMBEDDINGS_TABLES
    )

    # Try ZeRO-1 first (better efficiency)
    viable_z1 = calculate_viable_micro_batch_sizes(
        variant, gpu_mem, dp, layers_per_gpu, experts_per_gpu, zero_strategy=1
    )

    # Try ZeRO-2
    viable_z2 = calculate_viable_micro_batch_sizes(
        variant, gpu_mem, dp, layers_per_gpu, experts_per_gpu, zero_strategy=2
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


# Global variable for color control (can be set via command-line args)
USE_COLOR = True


def color_text(text, color_name):
    """
    Add ANSI color codes if color output enabled.

    Args:
        text: String to colorize
        color_name: Color name (green, yellow, red, cyan, reset)

    Returns:
        Colored text string with ANSI codes (if enabled)
    """
    if not USE_COLOR:
        return text

    colors = {
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'cyan': '\033[96m',
        'bold': '\033[1m',
        'reset': '\033[0m'
    }

    return f"{colors.get(color_name, '')}{text}{colors['reset']}"


def format_memory_value_with_color(value_gb, headroom_gb):
    """
    Format memory value with color coding based on headroom.

    Args:
        value_gb: Memory value in GB
        headroom_gb: Headroom remaining in GB

    Returns:
        Formatted string with color
    """
    formatted = f"{value_gb:>7.1f}G"

    if headroom_gb > 20:
        return color_text(formatted, 'green')
    elif headroom_gb > 10:
        return color_text(formatted, 'yellow')
    else:
        return color_text(formatted, 'red')


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


print("=" * 140)
print("SCENARIOS: MEMORY ANALYSIS")
print("=" * 140)

for hw in HARDWARE:
    total_gpus = hw["gpus"]
    gpu_mem = hw["mem_gb"]
    dp_effective = total_gpus // (TP * PP)
    dp = dp_effective // EP

    print(f"\n{'#' * 140}")
    print(f"# {hw['name']} ({hw['nodes']} nodes, {gpu_mem} GB/GPU)")
    print(f"# PP={PP}, TP={TP}, EP={EP}, DP={dp}")
    print(f"{'#' * 140}")

    print(
        f"\n  {'Variant':<10} {'Model':>8} {'Grad':>8} {'Optim':>8} {'Activ':>8} {'Buf':>8} {'TOTAL':>8} {'Headroom':>9} {'ZeRO':>6} {'micro':>6}"
    )
    print(
        f"  {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 9} {'-' * 6} {'-' * 6}"
    )

    # Track viable micros for summary
    variant_viable_micros = {}

    for v in VARIANTS:
        d = v["d"]
        layers = v["layers"]
        layers_per_gpu = layers // PP

        # Calculate model memory using shared function
        model_mem_gb = calculate_model_memory_gb(
            v,
            layers_per_gpu,
            EXPERTS_PER_GPU,
            PARAM_BYTES,
            VOCAB,
            LAYER_NORM,
            FFN_WEIGHT_MATRICES,
            NUM_EMBEDDINGS_TABLES,
        )

        # Try ZeRO-1 first (better efficiency)
        viable_z1 = calculate_viable_micro_batch_sizes(
            v, gpu_mem, dp, layers_per_gpu, EXPERTS_PER_GPU, zero_strategy=1
        )

        # Try ZeRO-2
        viable_z2 = calculate_viable_micro_batch_sizes(
            v, gpu_mem, dp, layers_per_gpu, EXPERTS_PER_GPU, zero_strategy=2
        )

        # Determine best ZeRO strategy and micro batch size
        if viable_z1:
            # ZeRO-1 fits - use it with largest viable micro
            best_zero = 1
            micro = max(viable_z1)
            viable_micros = viable_z1
        elif viable_z2:
            # ZeRO-2 fits - use it with largest viable micro
            best_zero = 2
            micro = max(viable_z2)
            viable_micros = viable_z2
        else:
            # Neither fits - OOM
            best_zero = -1
            micro = 1  # Dummy value for display
            viable_micros = []

        # Store viable micros for summary
        variant_viable_micros[v["name"]] = (viable_micros, best_zero)

        # Calculate memory breakdown for display with chosen strategy and micro
        if best_zero > 0:
            memory_breakdown = calculate_memory_for_micro(
                micro=micro,
                model_mem_gb=model_mem_gb,
                variant=v,
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

            grad_show = memory_breakdown["grad"]
            optim_show = memory_breakdown["optim"]
            act_total = memory_breakdown["activation"]
            buf_total = memory_breakdown["buffer"]
            total_best = memory_breakdown["total"]
            headroom = gpu_mem - total_best
            z_label = f"Z{best_zero}"
            micro_display = micro
        else:
            # OOM case
            grad_show = 0
            optim_show = 0
            act_total = 0
            buf_total = 0
            total_best = 999.9  # Dummy large value
            headroom = gpu_mem - total_best
            z_label = "OOM"
            micro_display = "N/A"

        print(
            f"  {v['name']:<10} {model_mem_gb:>7.1f}G {grad_show:>7.1f}G {optim_show:>7.1f}G {act_total:>7.1f}G {buf_total:>7.1f}G {total_best:>7.1f}G {headroom:>8.1f}G {z_label:>6} {micro_display:>6}"
        )

    # Print summary of viable micro batch sizes
    print(f"\n  Viable micro batch sizes per variant:")
    for v_name, (micros, zero) in variant_viable_micros.items():
        if micros:
            print(f"  {v_name}: {micros} (ZeRO-{zero})")
        else:
            print(f"  {v_name}: None (OOM)")
# ============================================================================
# PHASE 2: Batch size analysis for DP=512
# ============================================================================
print(f"\n\n{'=' * 140}")
print("BATCH SIZE ANALYSIS (DP=TOTAL_GPUS/(PP*TP*EP))")
print("=" * 140)
for hw in HARDWARE:
    total_gpus = hw["gpus"]
    dp = total_gpus // (TP * PP * EP)
    print(f"\n--- All micro/accum combinations ---")
    print(
        f"  {'micro':>6} {'accum':>6} {'Tok/batch':>12} {'Steps':>10} {'Assessment':>35}"
    )
    for micro in MICROS:
        for accum in GRAD_ACCUM_VALUES:
            tok = dp * micro * accum * SEQ_LEN
            steps = TOTAL_TOKENS / tok
            assess = ""
            if abs(tok - TOKENS_PER_BATCH) / TOKENS_PER_BATCH < MARGIN:
                assess = "=== MATCHES TARGET ==="
            elif abs(tok - TOKENS_PER_BATCH * 2) / (TOKENS_PER_BATCH * 2) < MARGIN:
                assess = "=== MATCHES 134.2M TARGET ==="
            elif LOWER_BOUND_OPTIM_RANGE < steps < UPPER_BOUND_OPTIM_RANGE:
                assess = "reasonable"
            elif LOWER_BOUND_LOW_RANGE < steps < UPPER_BOUND_LOW_RANGE:
                assess = "borderline (low steps)"
            elif LOWER_BOUND_HIGH_RANGE < steps < UPPER_BOUND_HIGH_RANGE:
                assess = "borderline (high steps)"
            elif steps >= MAX_STEPS:
                assess = "too many steps"
            elif steps < MIN_STEPS:
                assess = "too few steps"
            if tok < MAX_TOKENS_PER_BATCH:
                print(
                    f"  {micro:>6} {accum:>6} {tok / 1e6:>10.1f}M {steps:>10,.0f} {assess:>35}"
                )
# ============================================================================
# PHASE 3: Training time estimates
# ============================================================================
print(f"\n\n{'=' * 140}")
print("TRAINING TIME COMPARISON: ALL PLATFORMS AND VARIANTS")
print("=" * 140)

# Loop through all variants
for variant in VARIANTS:
    active_params_b = variant["active_params_B"]
    variant_name = variant["name"]

    print(f"\n{'#' * 140}")
    print(
        f"# Variant {variant_name} ({variant['total_params_B']:.1f}B total params, {active_params_b:.1f}B active params)"
    )
    print(f"{'#' * 140}")

    # Dynamically generate platform configurations from HARDWARE, ZERO_STRATEGY, and other variables
    platforms = []
    skipped_configs = []

    for hw in HARDWARE:
        total_gpus = hw["gpus"]
        peak_tflops = hw["peak_tflops_bf16"]
        gpu_mem = hw["mem_gb"]
        dp = total_gpus // (TP * PP * EP)

        # Calculate model memory and layers per GPU
        layers_per_gpu = variant["layers"] // PP

        # Try ZeRO-1 first (better efficiency)
        viable_z1 = calculate_viable_micro_batch_sizes(
            variant, gpu_mem, dp, layers_per_gpu, EXPERTS_PER_GPU, zero_strategy=1
        )

        # Try ZeRO-2
        viable_z2 = calculate_viable_micro_batch_sizes(
            variant, gpu_mem, dp, layers_per_gpu, EXPERTS_PER_GPU, zero_strategy=2
        )

        # Determine best ZeRO strategy and micro batch size
        if viable_z1:
            zero_config = ZERO_STRATEGY[0]  # ZeRO-1
            micro = max(viable_z1)  # Use largest viable
        elif viable_z2:
            zero_config = ZERO_STRATEGY[1]  # ZeRO-2
            micro = max(viable_z2)  # Use largest viable
        else:
            # OOM - skip this hardware for this variant
            skipped_configs.append(f"{hw['name']}")
            continue

        # Generate configurations for 1x and 2x target batch size
        for batch_multiplier in [1, 2]:
            tok_batch = TOKENS_PER_BATCH * batch_multiplier

            # Calculate required accumulation steps: tok_batch = dp * micro * accum * SEQ_LEN
            accum = int(tok_batch / (dp * micro * SEQ_LEN))

            # Calculate training steps
            steps = int(TOTAL_TOKENS / tok_batch)

            # Create platform name
            batch_label = (
                f"{tok_batch / 1e6:.0f}M"
                if batch_multiplier == 2
                else f"{tok_batch / 1e6:.1f}M"
            )
            platform_name = f"{hw['name']} ({batch_label})"

            platforms.append(
                {
                    "name": platform_name,
                    "gpus": total_gpus,
                    "zero": zero_config["zero"],
                    "eff": zero_config["eff"],
                    "micro": micro,
                    "accum": accum,
                    "tok_batch": tok_batch,
                    "steps": steps,
                    "peak_tflops": peak_tflops,
                }
            )

    # Calculate training times using FLOPs-based approach
    base = platforms[0]
    print(
        f"\n  {'Platform':<25} {'GPUs':>6} {'ZeRO':>5} {'micro':>6} {'accum':>6} {'Tok/batch':>10} {'Steps':>8} {'Rel Speed':>10} {'Est. Time':>12}"
    )
    print(
        f"  {'-' * 25} {'-' * 6} {'-' * 5} {'-' * 6} {'-' * 6} {'-' * 10} {'-' * 8} {'-' * 10} {'-' * 12}"
    )

    # Calculate baseline time for relative speed comparison
    base_time_months = calculate_training_time_months(
        TOTAL_TOKENS,
        active_params_b,
        base["gpus"],
        base["peak_tflops"],
        MFU,
        base["eff"],
    )

    for p in platforms:
        # Calculate absolute training time using FLOPs
        est_months = calculate_training_time_months(
            TOTAL_TOKENS, active_params_b, p["gpus"], p["peak_tflops"], MFU, p["eff"]
        )

        # Calculate relative speed compared to baseline
        rel_speed = base_time_months / est_months

        # Apply uncertainty range
        months_low = est_months * TIME_ESTIMATE_LOW_MULTIPLIER
        months_high = est_months * TIME_ESTIMATE_HIGH_MULTIPLIER

        print(
            f"  {p['name']:<25} {p['gpus']:>6} {'Z' + str(p['zero']):>5} {p['micro']:>6} {p['accum']:>6} {p['tok_batch'] / 1e6:>8.1f}M {p['steps']:>8,} {rel_speed:>9.1f}x {months_low:>4.1f}-{months_high:.1f}mo"
        )

    # Show skipped configurations if any
    if skipped_configs:
        print(f"\n  Note: Skipped hardware (OOM for this variant): {', '.join(skipped_configs)}")

# ============================================================================
# PHASE 4: ZeRO-2 overhead at DP=512
# ============================================================================
print(f"\n\n{'=' * 140}")
print("ZeRO-2 COMMUNICATION OVERHEAD ANALYSIS")
print("=" * 140)

for hw in HARDWARE:
    total_gpus = hw["gpus"]
    gpu_mem = hw["mem_gb"]
    inter_node_bw = hw["inter_node_bw_gb"]
    dp = total_gpus // (TP * PP * EP)

    # Calculate DP values to analyze: half DP and full DP
    dp_values = [dp // 2, dp]

    print(f"\n{'#' * 140}")
    print(f"# {hw['name']} (DP={dp}, inter-node BW={inter_node_bw} GB/s)")
    print(f"{'#' * 140}")

    for v in VARIANTS:
        d = v["d"]
        layers = v["layers"]
        layers_per_gpu = layers // PP

        # Calculate model memory using shared function
        model_mem_gb = calculate_model_memory_gb(
            v,
            layers_per_gpu,
            EXPERTS_PER_GPU,
            PARAM_BYTES,
            VOCAB,
            LAYER_NORM,
            FFN_WEIGHT_MATRICES,
            NUM_EMBEDDINGS_TABLES,
        )

        # Check which micro batch sizes from MICROS work with ZeRO-2
        viable_micros = calculate_viable_micro_batch_sizes(
            v, gpu_mem, dp, layers_per_gpu, EXPERTS_PER_GPU, zero_strategy=2
        )

        # Only analyze communication if at least one micro batch size works with ZeRO-2
        if not viable_micros:
            continue  # Skip variants that don't fit with ZeRO-2

        # Print variant header
        print(
            f"\n  Variant {v['name']} (d={d}, {layers} layers, model={model_mem_gb:.1f} GB/GPU, viable micro={viable_micros}):"
        )

        # Analyze communication for different DP values
        for dp_val in dp_values:
            rs_volume_mb = model_mem_gb * 2 / dp_val * 1000  # factor of 2 for gradients
            rs_time_ms = (
                model_mem_gb * 2 / dp_val / inter_node_bw * 1000
            )  # time at inter-node bandwidth
            print(
                f"    DP={dp_val}: reduce-scatter = {rs_volume_mb:.0f} MB/micro-batch, {rs_time_ms:.1f} ms at {inter_node_bw} GB/s"
            )
# ============================================================================
# PHASE 5: All-to-all volumes at lower micro-batch
# ============================================================================
print(f"\n\n{'=' * 140}")
print("ALL-TO-ALL COMMUNICATION VOLUMES (Intra-node)")
print("=" * 140)

for hw in HARDWARE:
    intra_node_bw = hw["intra_node_bw_gbps"]

    print(f"\n{'#' * 140}")
    print(f"# {hw['name']} (Intra-node BW={intra_node_bw} GB/s)")
    print(f"{'#' * 140}")

    for v in VARIANTS:
        d = v["d"]
        v_name = v["name"]

        print(f"\n  Variant {v_name} (d={d}):")
        for micro in MICROS:
            # All-to-all for MoE routing: activations sent to experts
            fwd_bytes = SEQ_LEN * micro * TOPK * d * PARAM_BYTES
            fwd_bwd_gb = (
                fwd_bytes * 2 / 1e9
            )  # factor of 2 for forward + backward passes
            a2a_time_ms = fwd_bwd_gb / intra_node_bw * 1000
            print(
                f"    micro={micro}: fwd+bwd = {fwd_bwd_gb:.2f} GB, time = {a2a_time_ms:.2f} ms at {intra_node_bw} GB/s"
            )
