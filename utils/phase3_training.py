"""
Phase 3: Training Time Estimation functions.

Generates platform configurations and estimates training times for model variants
across different hardware, comparing ZeRO strategies and batch sizes.
"""

# Import configurations
from configuration.project_config import *
from configuration.advanced_config import *

# Import core calculation functions
from .core_calculations import (
    calculate_viable_micro_batch_sizes,
    calculate_training_time_months
)

# Import formatting utilities
from .formatting_utils import color_text


def generate_platform_configs(variant, hw, dp, layers_per_gpu, experts_per_gpu):
    """
    Generate platform configurations for a variant/hardware combination.

    Tries ZeRO-1 first (best efficiency), falls back to ZeRO-2 if needed.
    Generates configs for both 1x and 2x target batch sizes.

    Args:
        variant: Variant dict with model architecture
        hw: Hardware dict with GPU specs
        dp: Data parallelism degree
        layers_per_gpu: Number of layers per GPU
        experts_per_gpu: Number of experts per GPU

    Returns:
        dict with:
        - platforms: list of platform config dicts
        - oom: bool indicating if variant doesn't fit
        - oom_reason: str explaining why OOM occurred
    """
    total_gpus = hw["gpus"]
    peak_tflops = hw["peak_tflops_bf16"]
    gpu_mem = hw["mem_gb"]

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
        zero_config = ZERO_STRATEGY[0]  # ZeRO-1
        micro = max(viable_z1)  # Use largest viable
    elif viable_z2:
        zero_config = ZERO_STRATEGY[1]  # ZeRO-2
        micro = max(viable_z2)  # Use largest viable
    else:
        # OOM - cannot fit on this hardware
        return {
            "platforms": [],
            "oom": True,
            "oom_reason": f"Insufficient memory even with ZeRO-2 (requires >{gpu_mem}GB)"
        }

    # Generate configurations for 1x and 2x target batch size
    platforms = []
    for batch_multiplier in [1, 2]:
        tok_batch = TOKENS_PER_BATCH * batch_multiplier

        # Calculate required accumulation steps: tok_batch = dp * micro * accum * SEQ_LEN
        accum = max(1, round(tok_batch / (dp * micro * SEQ_LEN)))

        # Calculate training steps
        steps = int(TOTAL_TOKENS / tok_batch)

        # Create platform name
        batch_label = (
            f"{tok_batch / 1e6:.0f}M"
            if batch_multiplier == 2
            else f"{tok_batch / 1e6:.1f}M"
        )
        platform_name = f"{hw['name']} ({batch_label})"

        platforms.append({
            "name": platform_name,
            "gpus": total_gpus,
            "zero": zero_config["zero"],
            "eff": zero_config["eff"],
            "micro": micro,
            "accum": accum,
            "tok_batch": tok_batch,
            "steps": steps,
            "peak_tflops": peak_tflops,
        })

    return {
        "platforms": platforms,
        "oom": False,
        "oom_reason": None
    }


def calculate_training_metrics(platform, active_params_b, base_time_months):
    """
    Calculate training time metrics for a platform configuration.

    Args:
        platform: Platform dict with gpus, peak_tflops, eff
        active_params_b: Active parameters in billions
        base_time_months: Baseline time for relative speed calculation

    Returns:
        dict with:
        - est_months: Estimated training time in months
        - rel_speed: Relative speed vs baseline (1.0x = baseline)
        - months_low: Lower bound with optimistic multiplier
        - months_high: Upper bound with pessimistic multiplier
        - efficiency_rating: Star rating (0-3)
    """
    # Calculate absolute training time using FLOPs
    est_months = calculate_training_time_months(
        TOTAL_TOKENS,
        active_params_b,
        platform["gpus"],
        platform["peak_tflops"],
        MFU,
        platform["eff"]
    )

    # Calculate relative speed compared to baseline
    rel_speed = base_time_months / est_months

    # Apply uncertainty range
    months_low = est_months * TIME_ESTIMATE_LOW_MULTIPLIER
    months_high = est_months * TIME_ESTIMATE_HIGH_MULTIPLIER

    # Assign efficiency rating (0=best, 3=worst)
    if rel_speed >= 2.0:
        efficiency_rating = 0  # ⭐⭐⭐
    elif rel_speed >= 1.5:
        efficiency_rating = 1  # ⭐⭐
    elif rel_speed >= 1.0:
        efficiency_rating = 2  # ⭐
    else:
        efficiency_rating = 3  # (no stars)

    return {
        "est_months": est_months,
        "rel_speed": rel_speed,
        "months_low": months_low,
        "months_high": months_high,
        "efficiency_rating": efficiency_rating
    }


def generate_training_recommendations(training_results):
    """
    Analyze training results and generate actionable recommendations.

    Args:
        training_results: dict with variant results and OOM cases

    Returns:
        list of recommendation strings
    """
    recommendations = []

    if not training_results.get("variants"):
        return ["⚠️  No training configurations were viable"]

    # Find fastest overall configuration
    fastest_config = None
    fastest_time = float('inf')
    fastest_variant = None

    for variant_name, platforms in training_results["variants"].items():
        for platform in platforms:
            if platform["est_months"] < fastest_time:
                fastest_time = platform["est_months"]
                fastest_config = platform
                fastest_variant = variant_name

    if fastest_config:
        recommendations.append(
            f"🏆 Fastest configuration: Variant {fastest_variant} on {fastest_config['name']} "
            f"({fastest_config['months_low']:.1f}-{fastest_config['months_high']:.1f} months, "
            f"{fastest_config['rel_speed']:.1f}x speed)"
        )

    # Find most cost-effective (best time per GPU)
    best_efficiency = None
    best_efficiency_ratio = float('inf')
    best_efficiency_variant = None

    for variant_name, platforms in training_results["variants"].items():
        for platform in platforms:
            efficiency_ratio = platform["est_months"] / (platform["gpus"] / 1000)  # months per 1000 GPUs
            if efficiency_ratio < best_efficiency_ratio:
                best_efficiency_ratio = efficiency_ratio
                best_efficiency = platform
                best_efficiency_variant = variant_name

    if best_efficiency and best_efficiency != fastest_config:
        recommendations.append(
            f"💰 Most cost-effective: Variant {best_efficiency_variant} on {best_efficiency['name']} "
            f"({best_efficiency['est_months'] / (best_efficiency['gpus'] / 1000):.2f} months per 1000 GPUs)"
        )

    # Analyze ZeRO strategy impact
    zero1_configs = []
    zero2_configs = []

    for variant_name, platforms in training_results["variants"].items():
        for platform in platforms:
            if platform["zero"] == 1:
                zero1_configs.append((variant_name, platform))
            elif platform["zero"] == 2:
                zero2_configs.append((variant_name, platform))

    if zero1_configs and zero2_configs:
        avg_z1_eff = sum(p["eff"] for _, p in zero1_configs) / len(zero1_configs)
        avg_z2_eff = sum(p["eff"] for _, p in zero2_configs) / len(zero2_configs)
        overhead_pct = ((avg_z1_eff - avg_z2_eff) / avg_z1_eff) * 100

        recommendations.append(
            f"⚡ ZeRO strategy impact: ZeRO-2 adds ~{overhead_pct:.0f}% overhead vs ZeRO-1 "
            f"({len(zero2_configs)} configs require ZeRO-2)"
        )

    # Check for OOM cases and suggest alternatives
    if training_results.get("oom_cases"):
        oom_variants = list(training_results["oom_cases"].keys())
        if oom_variants:
            recommendations.append(
                f"⚠️  Variants {', '.join(oom_variants)} have OOM issues on some hardware "
                f"→ Consider H200 ({H200_MEM_GB}GB) vs H100 ({H100_MEM_GB}GB) or reduce model size"
            )

    # Analyze batch size impact (1x vs 2x)
    batch_size_impacts = []
    for variant_name, platforms in training_results["variants"].items():
        # Group by hardware (first part of name before parentheses)
        hw_groups = {}
        for p in platforms:
            hw_name = p["name"].split("(")[0].strip()
            if hw_name not in hw_groups:
                hw_groups[hw_name] = []
            hw_groups[hw_name].append(p)

        # Compare 1x vs 2x for each hardware
        for hw_name, hw_platforms in hw_groups.items():
            if len(hw_platforms) >= 2:
                p1x = hw_platforms[0]  # First is 1x
                p2x = hw_platforms[1]  # Second is 2x
                time_diff_pct = ((p2x["est_months"] - p1x["est_months"]) / p1x["est_months"]) * 100
                batch_size_impacts.append((variant_name, hw_name, time_diff_pct))

    if batch_size_impacts:
        avg_impact = sum(abs(t[2]) for t in batch_size_impacts) / len(batch_size_impacts)
        if avg_impact > 5:  # More than 5% difference
            recommendations.append(
                f"📊 Batch size impact: 2x batch size changes training time by ~{avg_impact:.1f}% on average"
            )

    return recommendations


def export_training_results_csv(training_results, filename):
    """
    Export training time analysis results to CSV file.

    Args:
        training_results: dict with variant results
        filename: Output CSV filename
    """
    import csv

    with open(filename, 'w', newline='') as f:
        fieldnames = [
            'variant', 'platform', 'gpus', 'zero', 'micro', 'accum',
            'tok_batch', 'steps', 'est_months', 'rel_speed', 'months_low',
            'months_high', 'efficiency_rating'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for variant_name, platforms in training_results.get("variants", {}).items():
            for platform in platforms:
                writer.writerow({
                    'variant': variant_name,
                    'platform': platform['name'],
                    'gpus': platform['gpus'],
                    'zero': platform['zero'],
                    'micro': platform['micro'],
                    'accum': platform['accum'],
                    'tok_batch': platform['tok_batch'],
                    'steps': platform['steps'],
                    'est_months': platform.get('est_months', 0),
                    'rel_speed': platform.get('rel_speed', 0),
                    'months_low': platform.get('months_low', 0),
                    'months_high': platform.get('months_high', 0),
                    'efficiency_rating': platform.get('efficiency_rating', 3)
                })
