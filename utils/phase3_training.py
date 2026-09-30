"""
Phase 3: Training Time Estimation functions.

Generates platform configurations and estimates training times for model variants
across different hardware, comparing ZeRO strategies and batch sizes.
"""

import math

# Import configurations
from configuration.project_config import *
from configuration.advanced_config import *

# Import core calculation functions
from .core_calculations import (
    calculate_viable_micro_batch_sizes,
    calculate_training_time_months,
    calculate_training_time_months_with_overhead,
    compute_attention_flops_dim,
)

# Import communication models for overhead estimation
from .phase6_pp_comm import calculate_pp_communication
from .phase5_alltoall_comm import calculate_moe_all_to_all

# Import formatting utilities
from .formatting_utils import color_text


# ---------------------------------------------------------------------------
# Efficiency-rating thresholds.
#
# The rating is a position within a variant's OWN [slowest, fastest] range, on a
# log scale, not a fixed multiplier on rel_speed. Fixed multipliers cannot work
# here: measured over the real sweep, per-variant spread (slowest/fastest) runs
# from 32.5x (Llama-3.1-405B) to 20,642x (Llama-3.2-3B) -- a 635x difference in
# the scale being measured -- so any single multiplier threshold means something
# different for every variant and the ratings are not comparable across them.
#
# Positions are log-scaled because the est_months values within a variant are
# spread geometrically (GPU count doubles), so a linear position would put
# almost every config against the fast end.
#
# Thresholds chosen from the measured distribution over all 5,084 rows:
#   rating 0 (>= 0.80): 1,351 rows (26.6%)
#   rating 1 (>= 0.60): 1,834 rows (36.1%)
#   rating 2 (>= 0.40): 1,068 rows (21.0%)
#   rating 3 ( < 0.40):   831 rows (16.3%)
# Every band is populated and none dominates. The alternatives measured worse:
# 0.75/0.50/0.25 left rating 3 at 7.1%, and 0.70/0.45/0.20 left it at 5.2%.
EFFICIENCY_POSITION_EXCELLENT = 0.80  # rating 0 (***)
EFFICIENCY_POSITION_GOOD = 0.60       # rating 1 (**)
EFFICIENCY_POSITION_MODERATE = 0.40   # rating 2 (*)



def generate_platform_configs(variant, hw, dp, layers_per_gpu, experts_per_gpu):
    """
    Generate platform configurations for a variant/hardware combination.

    Tries ZeRO-1 first (best efficiency), falls back to ZeRO-2 if needed.
    Sweeps the full grid of viable micro batch sizes x configured
    GRAD_ACCUM_VALUES, deriving the global batch and step count from each
    (micro, accum) pair.

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

    # Peak throughput must follow PRECISION. Previously this always read
    # peak_tflops_bf16, so an FP8 run halved its memory while still computing
    # time at BF16 throughput -- overstating FP8 training time ~2x.
    # A10G (g5) is Ampere and has NO FP8 support, so peak_tflops_fp8 is 0 there
    # and an FP8 run on it is not a valid configuration.
    if PRECISION == "BF16":
        peak_tflops = hw["peak_tflops_bf16"]
    else:
        peak_tflops = hw.get("peak_tflops_fp8") or 0
        if not peak_tflops:
            if "peak_tflops_fp8" in hw:
                reason = (
                    f"{hw['name']} has no FP8 support (peak_tflops_fp8 is 0 -- "
                    f"A10G/Ampere lacks FP8 tensor cores)."
                )
            else:
                reason = (
                    f"{hw['name']} does not define peak_tflops_fp8, which is "
                    f"required when PRECISION is FP8. Add it to the hardware entry."
                )
            return {"platforms": [], "oom": True, "oom_reason": reason}
    gpu_mem = hw["mem_gb"]

    # Hybrid-attention models (Qwen3.5) interleave linear attention with full
    # attention: full_attention_interval=4 means layer_types repeats
    # [linear, linear, linear, full]. Linear attention is O(seq_len), so only
    # the full-attention layers belong in the O(seq_len^2) FLOPs term.
    # Absent or 1 => every layer is full attention (all other variants).
    interval = variant.get("full_attention_interval", 1) or 1
    full_attention_layers = variant["layers"] // interval if interval > 1 else variant["layers"]

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

    # Try ZeRO-3, which additionally shards the resident parameters. Reached only
    # when ZeRO-1 and ZeRO-2 both OOM: it fits strictly more, at 3*Psi per step
    # against their 2*Psi (see phase 4.2).
    viable_z3 = calculate_viable_micro_batch_sizes(
        variant, gpu_mem, dp, layers_per_gpu, experts_per_gpu, zero_strategy=3,
        tp=TP
    )

    # Determine best ZeRO strategy and viable micro batch sizes
    if viable_z1:
        zero_config = ZERO_STRATEGY[0]  # ZeRO-1
        viable_micros = viable_z1
    elif viable_z2:
        zero_config = ZERO_STRATEGY[1]  # ZeRO-2
        viable_micros = viable_z2
    elif viable_z3:
        zero_config = ZERO_STRATEGY[2]  # ZeRO-3
        viable_micros = viable_z3
    else:
        # OOM - cannot fit on this hardware
        return {
            "platforms": [],
            "oom": True,
            "oom_reason": f"Insufficient memory even with ZeRO-3 (requires >{gpu_mem}GB)"
        }

    # MoE layers on this GPU (for all-to-all comm overhead)
    moe_layers = variant["layers"] - variant.get("dense_layers", variant["layers"])
    total_layers = variant["layers"]
    moe_layers_per_gpu = (
        int(layers_per_gpu * moe_layers / total_layers) if total_layers > 0 else 0
    )

    # Sweep the full micro x accum grid. Gradient accumulation is an explicit
    # input the user controls (it affects convergence/perplexity), so we sweep
    # the configured GRAD_ACCUM_VALUES rather than deriving accum from a target.
    seq_len = variant.get("seq_len", SEQ_LEN)
    platforms = []
    for micro in viable_micros:
        for accum in GRAD_ACCUM_VALUES:
            gbs = dp * micro * accum  # global batch in sequences
            tok_batch = gbs * seq_len

            # Skip configurations that exceed the max tokens-per-batch cap
            if tok_batch > MAX_TOKENS_PER_BATCH:
                continue

            steps = int(TOTAL_TOKENS / tok_batch) if tok_batch > 0 else 0

            platform_name = f"{hw['name']} (m{micro}/a{accum}, {tok_batch / 1e6:.1f}M)"

            platforms.append({
                "name": platform_name,
                "gpus": total_gpus,
                "zero": zero_config["zero"],
                "eff": zero_config["eff"],
                "micro": micro,
                "accum": accum,
                "gbs": gbs,
                "tok_batch": tok_batch,
                "steps": steps,
                "peak_tflops": peak_tflops,
                # Inputs needed for overhead (bubble + exposed comm) estimation
                "dp": dp,
                "layers": variant["layers"],
                # Layers carrying FULL (quadratic) attention. Hybrid models
                # interleave linear attention, which is O(seq_len) and must not
                # enter the O(seq_len^2) term. full_attention_interval=4 means
                # 1 full layer per 4 (Qwen3.5); absent means every layer is full.
                "full_attention_layers": full_attention_layers,
                "layers_per_gpu": layers_per_gpu,
                "moe_layers_per_gpu": moe_layers_per_gpu,
                "pp": PP,
                "vp": VP,
                "ep": EP,
                "hidden_dim": variant["d"],
                # Width for the O(seq_len^2) attention FLOPs term ONLY. Distinct
                # from hidden_dim, which stays `d` for the PP and MoE comm models
                # because those move d-wide activation vectors. See
                # compute_attention_flops_dim.
                "attn_flops_dim": compute_attention_flops_dim(variant),
                "seq_len": seq_len,
                "inter_node_bw_gb": hw["inter_node_bw_gb"],
                "intra_node_bw_gbps": hw["intra_node_bw_gbps"],
                "nodes": hw["nodes"],
            })

    if not platforms:
        return {
            "platforms": [],
            "oom": True,
            "oom_reason": (
                f"No micro×accum combination stays under the "
                f"{MAX_TOKENS_PER_BATCH / 1e6:.0f}M tokens/batch cap"
            ),
        }

    return {
        "platforms": platforms,
        "oom": False,
        "oom_reason": None
    }


def calculate_platform_overhead(platform):
    """Estimate PP bubble fraction and exposed comm ms/step for a platform.

    Uses the Phase 6 (PP send/recv) and Phase 5 (MoE all-to-all) models,
    keyed on the platform's micro/gbs/parallelism so the result varies with
    the swept gradient-accumulation value.

    Returns:
        tuple: (bubble_fraction, exposed_comm_ms_per_step)
    """
    # GPUs per node comes from the hardware entry, not a fixed 8. It decides
    # whether PP (phase 6) and EP (phase 5) traffic is treated as intra-node
    # NVLink or inter-node EFA, and those differ by ~3 orders of magnitude, so
    # assuming 8 on a platform with another density (GB200 NVL is 4) picks the
    # wrong link. Fall back to 8 only when nodes is missing or zero.
    nodes = platform.get("nodes") or 0
    gpus_per_node = platform["gpus"] // nodes if nodes > 0 else 8

    pp_metrics = calculate_pp_communication(
        layers=platform["layers_per_gpu"] * platform["pp"],
        hidden_dim=platform["hidden_dim"],
        seq_len=platform["seq_len"],
        mbs=platform["micro"],
        gbs=platform["gbs"],
        pp=platform["pp"],
        vp=platform["vp"],
        dp=platform["dp"],
        ep=platform["ep"],
        dtype_bytes=PARAM_BYTES,
        inter_node_bw_gb=platform["inter_node_bw_gb"],
        intra_node_bw_gbps=platform["intra_node_bw_gbps"],
        gpus_per_node=gpus_per_node,
    )
    bubble_fraction = pp_metrics["pipeline_bubble_pct"] / 100.0
    exposed_comm_ms = pp_metrics["estimated_pp_time_ms"]

    # Exposed MoE all-to-all (only when the variant has MoE layers on-GPU)
    if platform["moe_layers_per_gpu"] > 0:
        a2a = calculate_moe_all_to_all(
            layers_per_gpu=platform["moe_layers_per_gpu"],
            hidden_dim=platform["hidden_dim"],
            seq_len=platform["seq_len"],
            mbs=platform["micro"],
            gbs=platform["gbs"],
            dp=platform["dp"],
            ep=platform["ep"],
            topk=TOPK,
            num_experts=N_EXPERTS,
            dtype_bytes=PARAM_BYTES,
            intra_node_bw_gbps=platform["intra_node_bw_gbps"],
            inter_node_bw_gb=platform["inter_node_bw_gb"],
            gpus_per_node=gpus_per_node,
        )
        exposed_comm_ms += A2A_EXPOSED_FRACTION * a2a["a2a_per_step_ms"]

    return bubble_fraction, exposed_comm_ms


def calculate_training_metrics(platform, active_params_b):
    """
    Calculate training time metrics for a platform configuration.

    Args:
        platform: Platform dict with gpus, peak_tflops, eff
        active_params_b: Active parameters in billions

    Returns:
        dict with:
        - est_months: Estimated training time in months (overhead-inclusive)
        - rel_speed: None -- set by classify_platform_efficiency
        - months_low: Lower bound with optimistic multiplier
        - months_high: Upper bound with pessimistic multiplier
        - bubble_pct: PP bubble as % of total time
        - comm_overhead_pct: Exposed comm as % of total time
        - efficiency_rating: None -- set by classify_platform_efficiency

    rel_speed and efficiency_rating are deliberately NOT assigned here. Both are
    relative to the variant's whole set of configurations -- rel_speed is a
    speed-up over the SLOWEST config, and the rating is a position within the
    [slowest, fastest] range -- so neither can be known while sweeping one
    config at a time. Call classify_platform_efficiency on the collected list.
    """
    # Estimate PP bubble + exposed comm for this specific (micro, accum) config
    bubble_fraction, exposed_comm_ms = calculate_platform_overhead(platform)

    overhead = calculate_training_time_months_with_overhead(
        TOTAL_TOKENS,
        active_params_b,
        platform["gpus"],
        platform["peak_tflops"],
        MFU,
        platform["eff"],
        platform["tok_batch"],
        bubble_fraction=bubble_fraction,
        exposed_comm_ms_per_step=exposed_comm_ms,
        # Only full-attention layers contribute the O(seq_len^2) term; linear
        # attention is O(seq_len) and is already covered by the 6ND term.
        layers=platform.get("full_attention_layers") or platform["layers"],
        hidden_dim=platform["attn_flops_dim"],
        seq_len=platform["seq_len"],
        attention_flops_factor=ATTENTION_FLOPS_FACTOR,
    )
    est_months = overhead["est_months"]

    # Apply uncertainty range
    months_low = est_months * TIME_ESTIMATE_LOW_MULTIPLIER
    months_high = est_months * TIME_ESTIMATE_HIGH_MULTIPLIER

    return {
        "est_months": est_months,
        "rel_speed": None,
        "months_low": months_low,
        "months_high": months_high,
        "bubble_pct": overhead["bubble_pct"],
        "comm_overhead_pct": overhead["comm_pct"],
        "efficiency_rating": None
    }


def classify_platform_efficiency(platform_results):
    """
    Assign rel_speed and efficiency_rating across one variant's configurations.

    rel_speed is anchored on the SLOWEST configuration in the set, so it reads as
    "how many times faster than the worst viable option". It is therefore always
    >= 1.0, and exactly 1.0 for the slowest config itself. The previous anchor was
    all_platforms[0] -- the first entry in sweep order, which is neither the
    fastest nor the slowest and measured mid-pack for all 22 variants. That made
    1.0x mean nothing in particular and left 78% of rows rated 0 purely by
    exceeding 2x an arbitrary entry.

    efficiency_rating is the config's log-scaled position within this variant's
    [slowest, fastest] range: 0.0 at the slowest, 1.0 at the fastest, banded by
    the EFFICIENCY_POSITION_* thresholds. It is NOT a threshold on rel_speed --
    see those constants for why a fixed multiplier cannot be comparable across
    variants whose spreads differ by 635x.

    Args:
        platform_results: list of platform dicts, each already carrying est_months

    Returns:
        The same list, with rel_speed and efficiency_rating filled in.

    A set with fewer than 2 distinct est_months values has no range to position
    within; every config is then both slowest and fastest, so all get rel_speed
    1.0 and rating 0.
    """
    if not platform_results:
        return platform_results

    times = [p["est_months"] for p in platform_results if p["est_months"] > 0]
    if not times:
        for p in platform_results:
            p["rel_speed"] = 0.0
            p["efficiency_rating"] = 3
        return platform_results

    slowest = max(times)
    fastest = min(times)

    # Log-span of the set. Zero when every config takes the same time.
    span = math.log(slowest / fastest) if fastest > 0 and slowest > fastest else 0.0

    for p in platform_results:
        est = p["est_months"]
        if est <= 0:
            p["rel_speed"] = 0.0
            p["efficiency_rating"] = 3
            continue

        rel_speed = slowest / est
        p["rel_speed"] = rel_speed

        if span > 0:
            position = math.log(rel_speed) / span
        else:
            # Degenerate set: nothing to rank against, so no config is penalised.
            position = 1.0

        if position >= EFFICIENCY_POSITION_EXCELLENT:
            p["efficiency_rating"] = 0
        elif position >= EFFICIENCY_POSITION_GOOD:
            p["efficiency_rating"] = 1
        elif position >= EFFICIENCY_POSITION_MODERATE:
            p["efficiency_rating"] = 2
        else:
            p["efficiency_rating"] = 3

    return platform_results


def generate_training_recommendations(training_results):
    """
    Analyze training results and generate actionable recommendations.

    Args:
        training_results: dict with variant results and OOM cases

    Returns:
        list of recommendation strings

    Raises:
        ValueError: if any configuration is still unclassified. rel_speed and
            efficiency_rating are assigned by a separate set-level pass, so a
            caller that forgets it would otherwise format None into the output.
    """
    recommendations = []

    if not training_results.get("variants"):
        return ["⚠️  No training configurations were viable"]

    for variant_name, platforms in training_results["variants"].items():
        unrated = sum(1 for p in platforms if p.get("rel_speed") is None
                      or p.get("efficiency_rating") is None)
        if unrated:
            raise ValueError(
                f"{variant_name}: {unrated} configuration(s) have no rel_speed or "
                f"efficiency_rating. Call classify_platform_efficiency on each "
                f"variant's platform list before generating recommendations."
            )

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
            f"{fastest_config['rel_speed']:.1f}x faster than the slowest "
            f"{fastest_variant} option)"
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

    # Analyze gradient-accumulation impact on pipeline bubble and time.
    # Group by (variant, hardware) and compare across the swept accum values.
    accum_impacts = []  # (variant, hw, min_bubble, max_bubble, time_spread_pct)
    for variant_name, platforms in training_results["variants"].items():
        hw_groups = {}
        for p in platforms:
            hw_name = p["name"].split("(")[0].strip()
            hw_groups.setdefault(hw_name, []).append(p)

        for hw_name, hw_platforms in hw_groups.items():
            if len(hw_platforms) < 2:
                continue
            bubbles = [p.get("bubble_pct", 0) for p in hw_platforms]
            times = [p["est_months"] for p in hw_platforms]
            min_t, max_t = min(times), max(times)
            time_spread_pct = ((max_t - min_t) / min_t * 100) if min_t > 0 else 0
            accum_impacts.append(
                (variant_name, hw_name, min(bubbles), max(bubbles), time_spread_pct)
            )

    if accum_impacts:
        avg_spread = sum(t[4] for t in accum_impacts) / len(accum_impacts)
        max_bubble = max((t[3] for t in accum_impacts), default=0)
        if max_bubble > 1.0:
            recommendations.append(
                f"📊 Gradient accumulation impact: sweeping accum changes est. time by "
                f"~{avg_spread:.1f}% on average; higher accum reduces the pipeline "
                f"bubble (peak observed {max_bubble:.1f}%). Larger accum also affects "
                f"convergence/perplexity — tune deliberately."
            )
        else:
            recommendations.append(
                f"📊 Gradient accumulation impact: sweeping accum changes est. time by "
                f"~{avg_spread:.1f}% on average (PP bubble negligible at PP=1). Accum "
                f"still affects convergence/perplexity — tune deliberately."
            )

    return recommendations


def export_training_results_csv(training_results, filename):
    """
    Export training time analysis results to CSV file.

    Args:
        training_results: dict with variant results
        filename: Output CSV filename

    rel_speed and efficiency_rating export as EMPTY when unset, not as 0 and 3.
    Those two values are legitimate readings -- 3 is the worst rating -- so
    defaulting to them would make a config that was never classified
    indistinguishable from one genuinely rated worst.
    """
    import csv

    with open(filename, 'w', newline='') as f:
        fieldnames = [
            'variant', 'platform', 'gpus', 'zero', 'micro', 'accum', 'seq_len', 'gbs',
            'tok_batch', 'steps', 'est_months', 'rel_speed', 'months_low',
            'months_high', 'bubble_pct', 'comm_overhead_pct', 'efficiency_rating'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for variant_name, platforms in training_results.get("variants", {}).items():
            for platform in platforms:
                rel_speed = platform.get('rel_speed')
                rating = platform.get('efficiency_rating')
                writer.writerow({
                    'variant': variant_name,
                    'platform': platform['name'],
                    'gpus': platform['gpus'],
                    'zero': platform['zero'],
                    'micro': platform['micro'],
                    'accum': platform['accum'],
                    'seq_len': platform.get('seq_len', ''),
                    'gbs': platform.get('gbs', ''),
                    'tok_batch': platform['tok_batch'],
                    'steps': platform['steps'],
                    'est_months': platform.get('est_months', 0),
                    'rel_speed': '' if rel_speed is None else rel_speed,
                    'months_low': platform.get('months_low', 0),
                    'months_high': platform.get('months_high', 0),
                    'bubble_pct': platform.get('bubble_pct', 0),
                    'comm_overhead_pct': platform.get('comm_overhead_pct', 0),
                    'efficiency_rating': '' if rating is None else rating
                })
