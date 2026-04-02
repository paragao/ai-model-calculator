#!/usr/bin/python3

# Import configurations
from variants_config import VARIANTS
from hardware_config import HARDWARE
from project_config import *
from advanced_config import *

# Import validation
from utils.validation import validate_configuration

# Import core calculation functions
from utils.core_calculations import (
    calculate_training_time_months
)

# Import formatting utilities
from utils.formatting_utils import (
    USE_COLOR,
    color_text,
    format_memory_value_with_color,
    format_batch_assessment_with_color,
    format_training_time_with_color,
    format_communication_time_with_color,
    format_zero1_time_with_color,
    format_alltoall_time_with_color
)

# Import Phase 1: Memory Analysis
from utils.phase1_memory import (
    analyze_variant_memory,
    generate_recommendations,
    export_results_csv,
    export_results_json
)

# Import Phase 2: Batch Configuration
from utils.phase2_batch import (
    analyze_batch_configuration,
    generate_batch_recommendations,
    export_batch_results_csv
)

# Import Phase 3: Training Time
from utils.phase3_training import (
    generate_platform_configs,
    calculate_training_metrics,
    generate_training_recommendations,
    export_training_results_csv
)

# Import Phase 4: ZeRO-2 Communication
from utils.phase4_zero2_comm import (
    analyze_communication_overhead,
    generate_communication_recommendations,
    export_communication_results_csv
)

# Import Phase 4.1: ZeRO-1 Communication
from utils.phase4_1_zero1_comm import (
    analyze_zero1_communication_overhead,
    generate_zero1_recommendations,
    export_zero1_results_csv
)

# Import Phase 5: All-to-All Communication
from utils.phase5_alltoall_comm import (
    analyze_alltoall_communication,
    generate_alltoall_recommendations,
    export_alltoall_results_csv
)

# Validate configuration
validate_configuration()

# Note: All function definitions have been moved to phase-specific modules.
# This file now contains only the execution flow for all 5 analysis phases.

# ============================================================================
# PHASE 1: Memory analysis scenario
# ============================================================================
print("=" * 140)
print("SCENARIOS: MEMORY ANALYSIS")
print("=" * 140)

# Collect all results for analysis and export
all_results = {
    "hardware": [],
    "variants": {v["name"]: [] for v in VARIANTS}
}

for hw in HARDWARE:
    total_gpus = hw["gpus"]
    gpu_mem = hw["mem_gb"]
    dp = total_gpus // (TP * PP * CP)

    all_results["hardware"].append(hw["name"])

    print(f"\n{'#' * 140}")
    print(f"# {hw['name']} ({hw['nodes']} nodes, {gpu_mem} GB/GPU)")
    print(f"# PP={PP}, TP={TP}, EP={EP}, DP={dp}")
    print(f"{'#' * 140}")

    print(
        f"\n  {'Variant':<10} {'Model':>8} {'Grad':>8} {'Optim':>8} {'Activ':>8} {'Buf':>8} {'TOTAL':>8} {'Headroom':>9} {'Util%':>6} {'ZeRO':>6} {'micro':>6}"
    )
    print(
        f"  {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 9} {'-' * 6} {'-' * 6} {'-' * 6}"
    )

    for v in VARIANTS:
        layers_per_gpu = v["layers"] // PP

        # Use the new analyze_variant_memory function
        result = analyze_variant_memory(v, hw, dp, layers_per_gpu, EXPERTS_PER_GPU)

        # Store result for later analysis and export
        all_results["variants"][v["name"]].append(result)

        # Format and display row
        if result["best_zero"] > 0:
            # Success case
            mb = result["memory_breakdown"]
            em = result["efficiency_metrics"]

            # Format values with color
            headroom = em["wasted_headroom"]
            headroom_str = format_memory_value_with_color(headroom, headroom)
            util_pct = f"{em['utilization_pct']:>5.0f}%"

            z_label = f"Z{result['best_zero']}"
            micro_display = result["best_micro"]

            print(
                f"  {v['name']:<10} {mb['model']:>7.1f}G {mb['grad']:>7.1f}G {mb['optim']:>7.1f}G "
                f"{mb['activation']:>7.1f}G {mb['buffer']:>7.1f}G {mb['total']:>7.1f}G "
                f"{headroom_str} {util_pct:>6} {z_label:>6} {micro_display:>6}"
            )
        else:
            # OOM case
            oom_diag = result["oom_diagnostics"]
            model_mem = result["model_mem_gb"]

            # Show required vs available
            required = oom_diag["required_memory"]
            shortage = oom_diag["shortage"]

            z_label = color_text("OOM", "red")
            micro_display = "N/A"

            print(
                f"  {v['name']:<10} {model_mem:>7.1f}G {'---':>7} {'---':>7} {'---':>7} {'---':>7} "
                f"{color_text(f'{required:>7.1f}G', 'red')} {color_text(f'{-shortage:>8.1f}G', 'red')} "
                f"{'OOM':>6} {z_label:>6} {micro_display:>6}"
            )

    # Print summary of viable micro batch sizes
    print(f"\n  Viable micro batch sizes per variant:")
    for v_name in [v["name"] for v in VARIANTS]:
        result = all_results["variants"][v_name][-1]  # Latest result for this hardware
        if result["best_zero"] > 0:
            z1 = result["viable_micros_z1"]
            z2 = result["viable_micros_z2"]
            best_zero = result["best_zero"]

            if z1:
                print(f"  {v_name}: {z1} (ZeRO-1)")
            else:
                print(f"  {v_name}: {z2} (ZeRO-2)")
        else:
            print(f"  {v_name}: {color_text('None (OOM)', 'red')}")

# Generate and display recommendations
print(f"\n{'=' * 140}")
print("💡 RECOMMENDATIONS")
print(f"{'=' * 140}")
recommendations = generate_recommendations(all_results)
if recommendations:
    for rec in recommendations:
        print(f"  {rec}")
else:
    print("  ✅ All configurations look good!")

# Export Phase 1 results
export_results_csv(all_results, "phase1_memory_results.csv")
export_results_json(all_results, "phase1_memory_results.json")
print(f"\n📊 Phase 1 results exported to: phase1_memory_results.csv, phase1_memory_results.json")

# ============================================================================
# PHASE 2: Batch size analysis
# ============================================================================
print(f"\n\n{'=' * 140}")
print("BATCH SIZE ANALYSIS (DP=TOTAL_GPUS/(PP*TP*EP))")
print("=" * 140)

# Collect batch results for recommendations
batch_results = {}

for hw in HARDWARE:
    total_gpus = hw["gpus"]
    dp = total_gpus // (TP * PP * CP)

    print(f"\n{'#' * 140}")
    print(f"# {hw['name']}: DP={dp} (analyzing {len(MICROS)} micro × {len(GRAD_ACCUM_VALUES)} accum = {len(MICROS) * len(GRAD_ACCUM_VALUES)} combinations)")
    print(f"{'#' * 140}")

    print(
        f"\n  {'micro':>6} {'accum':>6} {'Tok/batch':>12} {'Steps':>10} {'Priority':>10} {'Assessment':>35}"
    )
    print(
        f"  {'-' * 6} {'-' * 6} {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 35}"
    )

    hw_configs = []

    for micro in MICROS:
        for accum in GRAD_ACCUM_VALUES:
            # Analyze this configuration
            config = analyze_batch_configuration(hw, micro, accum, dp)

            # Only display if tokens per batch is reasonable
            if config["tokens_per_batch"] < MAX_TOKENS_PER_BATCH:
                hw_configs.append(config)

                # Format assessment with color
                colored_assessment = format_batch_assessment_with_color(
                    config["assessment"], config["priority"]
                )

                # Priority indicator
                priority_symbols = ["⭐⭐⭐", "⭐⭐ ", "⭐  ", "   "]
                priority_str = priority_symbols[config["priority"]]

                print(
                    f"  {config['micro']:>6} {config['accum']:>6} "
                    f"{config['tokens_per_batch'] / 1e6:>10.1f}M {config['steps']:>10,.0f} "
                    f"{priority_str:>10} {colored_assessment:>35}"
                )

    # Store results for this hardware
    batch_results[hw["name"]] = hw_configs

    # Summary for this hardware
    optimal_count = len([c for c in hw_configs if c["priority"] == 0])
    good_count = len([c for c in hw_configs if c["priority"] <= 1])

    print(f"\n  Summary: {optimal_count} optimal, {good_count} good out of {len(hw_configs)} feasible configurations")

# Generate and display batch recommendations
print(f"\n{'=' * 140}")
print("💡 BATCH SIZE RECOMMENDATIONS")
print(f"{'=' * 140}")
batch_recommendations = generate_batch_recommendations(batch_results)
if batch_recommendations:
    for rec in batch_recommendations:
        print(f"  {rec}")
else:
    print("  ✅ All hardware has optimal batch configurations!")

# Export Phase 2 results
export_batch_results_csv(batch_results, "phase2_batch_results.csv")
print(f"\n📊 Phase 2 results exported to: phase2_batch_results.csv")

# ============================================================================
# PHASE 3: Training time estimates
# ============================================================================
print(f"\n\n{'=' * 140}")
print("TRAINING TIME COMPARISON: ALL PLATFORMS AND VARIANTS")
print("=" * 140)

# Collect results for recommendations
training_results = {
    "variants": {},  # variant_name -> list of platform results
    "oom_cases": {},  # variant_name -> list of OOM hardware names
}

# Loop through all variants
for variant in VARIANTS:
    active_params_b = variant["active_params_B"]
    variant_name = variant["name"]

    print(f"\n{'#' * 140}")
    print(
        f"# Variant {variant_name} ({variant['total_params_B']:.1f}B total params, {active_params_b:.1f}B active params)"
    )
    print(f"{'#' * 140}")

    # Generate platform configurations for this variant
    all_platforms = []
    oom_list = []

    for hw in HARDWARE:
        dp = hw["gpus"] // (TP * PP * CP)
        layers_per_gpu = variant["layers"] // PP

        config_result = generate_platform_configs(
            variant, hw, dp, layers_per_gpu, EXPERTS_PER_GPU
        )

        if config_result["oom"]:
            oom_list.append(hw["name"])
        else:
            all_platforms.extend(config_result["platforms"])

    # Calculate and display training times if platforms are available
    if all_platforms:
        # Calculate baseline time for relative speed comparison
        base = all_platforms[0]
        base_time_months = calculate_training_time_months(
            TOTAL_TOKENS,
            active_params_b,
            base["gpus"],
            base["peak_tflops"],
            MFU,
            base["eff"],
        )

        # Print table header
        print(
            f"\n  {'Platform':<25} {'GPUs':>6} {'ZeRO':>5} {'micro':>6} {'accum':>6} {'Tok/batch':>10} {'Steps':>8} {'Rel Speed':>10} {'Est. Time':>12}"
        )
        print(
            f"  {'-' * 25} {'-' * 6} {'-' * 5} {'-' * 6} {'-' * 6} {'-' * 10} {'-' * 8} {'-' * 10} {'-' * 12}"
        )

        # Calculate and display metrics for each platform
        platform_results = []
        for p in all_platforms:
            metrics = calculate_training_metrics(p, active_params_b, base_time_months)
            p.update(metrics)
            platform_results.append(p)

            # Format time string with color coding
            time_str = format_training_time_with_color(
                metrics["months_low"], metrics["months_high"], metrics["rel_speed"]
            )

            print(
                f"  {p['name']:<25} {p['gpus']:>6} {'Z' + str(p['zero']):>5} "
                f"{p['micro']:>6} {p['accum']:>6} {p['tok_batch'] / 1e6:>8.1f}M "
                f"{p['steps']:>8,} {metrics['rel_speed']:>9.1f}x {time_str}"
            )

        training_results["variants"][variant_name] = platform_results

    # Store and display OOM cases
    if oom_list:
        training_results["oom_cases"][variant_name] = oom_list
        print(f"\n  Note: Skipped hardware (OOM): {', '.join(oom_list)}")

# Generate and display recommendations
print(f"\n{'=' * 140}")
print("💡 TRAINING TIME RECOMMENDATIONS")
print(f"{'=' * 140}")
training_recommendations = generate_training_recommendations(training_results)
if training_recommendations:
    for rec in training_recommendations:
        print(f"  {rec}")
else:
    print("  ✅ All training configurations look good!")

# Export Phase 3 results
export_training_results_csv(training_results, "phase3_training_results.csv")
print(f"\n📊 Phase 3 results exported to: phase3_training_results.csv")

# ============================================================================
# PHASE 4: ZeRO-2 Communication Overhead Analysis
# ============================================================================
print(f"\n\n{'=' * 140}")
print("ZeRO-2 COMMUNICATION OVERHEAD ANALYSIS")
print("=" * 140)

# Collect results for recommendations
comm_results = {
    "hardware": {},  # hw_name -> {variant_name -> analysis results}
}

for hw in HARDWARE:
    total_gpus = hw["gpus"]
    inter_node_bw = hw["inter_node_bw_gb"]
    dp = total_gpus // (TP * PP * CP)

    # Generate DP values to analyze: multiple DP options from low to high
    # Start from minimum of 64 or dp//8, then double until reaching full DP
    dp_values = []
    dp_test = max(64, dp // 8)
    while dp_test <= dp:
        dp_values.append(dp_test)
        dp_test *= 2
    # Ensure full DP is included
    if dp not in dp_values:
        dp_values.append(dp)

    print(f"\n{'#' * 140}")
    print(f"# {hw['name']} (DP={dp}, inter-node BW={inter_node_bw} GB/s)")
    print(f"{'#' * 140}")
    print(f"  Analyzing DP values: {dp_values}")

    # Store hardware results
    hw_results = {}

    for v in VARIANTS:
        d = v["d"]
        layers = v["layers"]
        layers_per_gpu = layers // PP

        # Analyze communication overhead for this variant
        analysis = analyze_communication_overhead(
            v, hw, dp_values, layers_per_gpu, EXPERTS_PER_GPU
        )

        # Store results
        hw_results[v["name"]] = analysis

        # Only display if variant fits with ZeRO-2
        if not analysis["viable"]:
            continue

        # Print variant header
        print(
            f"\n  Variant {v['name']} (d={d}, {layers} layers, "
            f"model={analysis['model_mem_gb']:.1f} GB/GPU, "
            f"viable micro={analysis['viable_micros']}):"
        )

        # Display communication metrics for each DP value
        for metrics in analysis["comm_metrics"]:
            time_str = format_communication_time_with_color(
                metrics["rs_time_ms"], metrics["overhead_rating"]
            )
            print(
                f"    DP={metrics['dp']:>4}: reduce-scatter = {metrics['rs_volume_mb']:>5.0f} MB/micro-batch, {time_str} at {inter_node_bw} GB/s"
            )

    comm_results["hardware"][hw["name"]] = hw_results

# Generate and display recommendations
print(f"\n{'=' * 140}")
print("💡 ZeRO-2 COMMUNICATION RECOMMENDATIONS")
print(f"{'=' * 140}")
comm_recommendations = generate_communication_recommendations(comm_results)
if comm_recommendations:
    for rec in comm_recommendations:
        print(f"  {rec}")
else:
    print("  ✅ All ZeRO-2 communication patterns look optimal!")

# Export Phase 4 results
export_communication_results_csv(comm_results, "phase4_zero2_comm_results.csv")
print(f"\n📊 Phase 4 (ZeRO-2) results exported to: phase4_zero2_comm_results.csv")

# ============================================================================
# PHASE 4.1: ZeRO-1 Communication Overhead Analysis
# ============================================================================
print(f"\n\n{'=' * 140}")
print("ZeRO-1 COMMUNICATION OVERHEAD ANALYSIS")
print("=" * 140)

# Collect results for recommendations
zero1_results = {
    "hardware": {},  # hw_name -> {variant_name -> analysis results}
}

for hw in HARDWARE:
    total_gpus = hw["gpus"]
    inter_node_bw = hw["inter_node_bw_gb"]
    dp = total_gpus // (TP * PP * CP)

    # Generate DP values to analyze: multiple DP options from low to high
    # Start from minimum of 64 or dp//8, then double until reaching full DP
    dp_values = []
    dp_test = max(64, dp // 8)
    while dp_test <= dp:
        dp_values.append(dp_test)
        dp_test *= 2
    # Ensure full DP is included
    if dp not in dp_values:
        dp_values.append(dp)

    print(f"\n{'#' * 140}")
    print(f"# {hw['name']} (DP={dp}, inter-node BW={inter_node_bw} GB/s)")
    print(f"{'#' * 140}")
    print(f"  Analyzing DP values: {dp_values}")

    # Store hardware results
    hw_results = {}

    for v in VARIANTS:
        d = v["d"]
        layers = v["layers"]
        layers_per_gpu = layers // PP

        # Analyze ZeRO-1 communication overhead for this variant
        analysis = analyze_zero1_communication_overhead(
            v, hw, dp_values, layers_per_gpu, EXPERTS_PER_GPU
        )

        # Store results
        hw_results[v["name"]] = analysis

        # Only display if variant fits with ZeRO-1
        if not analysis["viable"]:
            continue

        # Print variant header
        print(
            f"\n  Variant {v['name']} (d={d}, {layers} layers, "
            f"model={analysis['model_mem_gb']:.1f} GB/GPU, "
            f"viable micro={analysis['viable_micros']}):"
        )

        # Display communication metrics for each DP value
        for metrics in analysis["comm_metrics"]:
            time_str = format_zero1_time_with_color(
                metrics["ag_time_ms"], metrics["overhead_rating"]
            )
            print(
                f"    DP={metrics['dp']:>4}: all-gather = {metrics['ag_volume_mb']:>5.0f} MB/micro-batch, {time_str} at {inter_node_bw} GB/s"
            )

    zero1_results["hardware"][hw["name"]] = hw_results

    # Check if any variants fit with ZeRO-1 on this hardware
    any_viable = any(result["viable"] for result in hw_results.values())
    if not any_viable:
        print(f"  ⚠️  No variants fit with ZeRO-1 on this hardware ({hw['mem_gb']} GB/GPU insufficient) - ZeRO-2 required")

# Generate and display recommendations
print(f"\n{'=' * 140}")
print("💡 ZeRO-1 COMMUNICATION RECOMMENDATIONS")
print(f"{'=' * 140}")
zero1_recommendations = generate_zero1_recommendations(zero1_results)
if zero1_recommendations:
    for rec in zero1_recommendations:
        print(f"  {rec}")
else:
    print("  ✅ All ZeRO-1 communication patterns look optimal!")

# Export Phase 4.1 results
export_zero1_results_csv(zero1_results, "phase4_1_zero1_comm_results.csv")
print(f"\n📊 Phase 4.1 (ZeRO-1) results exported to: phase4_1_zero1_comm_results.csv")

# ============================================================================
# PHASE 5: All-to-all Communication Volumes (Intra-node)
# ============================================================================
print(f"\n\n{'=' * 140}")
print("ALL-TO-ALL COMMUNICATION VOLUMES (Intra-node)")
print("=" * 140)

# Collect results for recommendations
a2a_results = {
    "hardware": {},  # hw_name -> list of variant analysis results
}

for hw in HARDWARE:
    intra_node_bw = hw["intra_node_bw_gbps"]

    print(f"\n{'#' * 140}")
    print(f"# {hw['name']} (Intra-node BW={intra_node_bw} GB/s)")
    print(f"{'#' * 140}")

    # Store hardware results
    hw_results = []

    for v in VARIANTS:
        # Analyze all-to-all communication for this variant
        analysis = analyze_alltoall_communication(v, hw, MICROS)
        hw_results.append(analysis)

        # Display variant header
        print(f"\n  Variant {analysis['variant_name']} (d={analysis['d']}):")

        # Display metrics for each micro batch size
        for metrics in analysis["a2a_metrics"]:
            time_str = format_alltoall_time_with_color(
                metrics["a2a_time_ms"], metrics["performance_rating"]
            )
            print(
                f"    micro={metrics['micro']}: fwd+bwd = {metrics['fwd_bwd_gb']:>4.2f} GB, time = {time_str} at {intra_node_bw} GB/s"
            )

    a2a_results["hardware"][hw["name"]] = hw_results

# Generate and display recommendations
print(f"\n{'=' * 140}")
print("💡 ALL-TO-ALL COMMUNICATION RECOMMENDATIONS")
print(f"{'=' * 140}")
a2a_recommendations = generate_alltoall_recommendations(a2a_results)
if a2a_recommendations:
    for rec in a2a_recommendations:
        print(f"  {rec}")
else:
    print("  ✅ All all-to-all communication patterns look optimal!")

# Export Phase 5 results
export_alltoall_results_csv(a2a_results, "phase5_alltoall_comm_results.csv")
print(f"\n📊 Phase 5 results exported to: phase5_alltoall_comm_results.csv")

# Summary of all exports
print(f"\n{'=' * 140}")
print("📊 EXPORT SUMMARY")
print(f"{'=' * 140}")
print("All analysis results have been exported to CSV/JSON files:")
print("  ✅ phase1_memory_results.csv / .json")
print("  ✅ phase2_batch_results.csv")
print("  ✅ phase3_training_results.csv")
print("  ✅ phase4_zero2_comm_results.csv")
print("  ✅ phase4_1_zero1_comm_results.csv")
print("  ✅ phase5_alltoall_comm_results.csv")
