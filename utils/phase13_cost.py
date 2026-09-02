"""
Phase 13: Cost Efficiency Analysis for Inference.

Calculates cost metrics for inference serving across different hardware
configurations and concurrency levels.

Key formulas:
  tokens_per_dollar = total_tok_s * 3600 / hourly_cost
  cost_per_million_tokens = hourly_cost / (total_tok_s * 3.6)
  tokens_per_sec_per_gpu = total_tok_s / num_gpus_used

Notes:
  - Pricing from INSTANCE_PRICING in inference_config.py.
  - Cost efficiency improves with higher concurrency (batching amortizes
    model weight reads across more requests).
  - Break-even analysis finds minimum concurrency to hit a target cost/M.
"""

import math
import csv


def calculate_cost_efficiency(total_tok_s, num_gpus, hourly_cost):
    """Cost metrics for inference serving.

    Formulas:
        tokens_per_dollar = total_tok_s * 3600 / hourly_cost
        cost_per_million_tokens = hourly_cost / (total_tok_s * 3.6)
        tokens_per_sec_per_gpu = total_tok_s / num_gpus

    Args:
        total_tok_s: Aggregate throughput in tokens per second.
        num_gpus: Number of GPUs used for serving.
        hourly_cost: On-demand hourly cost in USD.

    Returns:
        Dict with:
        - tokens_per_dollar: Tokens generated per dollar spent.
        - cost_per_million_tokens: USD cost per million output tokens.
        - tokens_per_sec_per_gpu: Per-GPU throughput.
        - total_tok_s: Aggregate throughput (echo back).
        - num_gpus: Number of GPUs (echo back).
        - hourly_cost: Hourly cost (echo back).
    """
    if total_tok_s <= 0 or hourly_cost <= 0:
        return {
            "tokens_per_dollar": 0.0,
            "cost_per_million_tokens": float("inf"),
            "tokens_per_sec_per_gpu": 0.0,
            "total_tok_s": total_tok_s,
            "num_gpus": num_gpus,
            "hourly_cost": hourly_cost,
        }

    # tokens_per_dollar: tokens generated in 1 hour / cost of 1 hour
    tokens_per_dollar = total_tok_s * 3600.0 / hourly_cost

    # cost_per_million_tokens: hourly_cost / (tokens per hour / 1e6)
    # = hourly_cost / (total_tok_s * 3600 / 1e6)
    # = hourly_cost / (total_tok_s * 3.6e-3)
    # Simplified: hourly_cost / (total_tok_s * 3.6)
    cost_per_million = hourly_cost / (total_tok_s * 3.6)

    tokens_per_gpu = total_tok_s / num_gpus if num_gpus > 0 else 0.0

    return {
        "tokens_per_dollar": round(tokens_per_dollar, 2),
        "cost_per_million_tokens": round(cost_per_million, 4),
        "tokens_per_sec_per_gpu": round(tokens_per_gpu, 2),
        "total_tok_s": total_tok_s,
        "num_gpus": num_gpus,
        "hourly_cost": hourly_cost,
    }


def calculate_breakeven_concurrency(hourly_cost, target_cost_per_million,
                                    throughput_at_concurrency_fn):
    """Find minimum concurrency to hit target cost per million tokens.

    Uses binary search over concurrency from 1 to 1024 to find the
    lowest concurrency where cost_per_million <= target.

    Formula (derived):
        cost_per_million = hourly_cost / (total_tok_s * 3.6)
        Need: total_tok_s >= hourly_cost / (target_cost_per_million * 3.6)

    Args:
        hourly_cost: On-demand hourly cost in USD.
        target_cost_per_million: Target cost per million tokens in USD.
        throughput_at_concurrency_fn: Callable(concurrency) -> total_tok_s.
            Returns aggregate throughput at given concurrency level.

    Returns:
        Dict with:
        - found: True if a break-even concurrency was found.
        - breakeven_concurrency: Minimum concurrency (or None if not found).
        - required_tok_s: Minimum throughput needed.
        - achieved_tok_s: Throughput at break-even concurrency.
        - actual_cost_per_million: Actual cost at break-even point.
    """
    if hourly_cost <= 0 or target_cost_per_million <= 0:
        return {
            "found": False,
            "breakeven_concurrency": None,
            "required_tok_s": 0.0,
            "achieved_tok_s": 0.0,
            "actual_cost_per_million": float("inf"),
        }

    # Required throughput: cost_per_million = hourly / (tok_s * 3.6)
    # Solving: tok_s >= hourly / (target * 3.6)
    required_tok_s = hourly_cost / (target_cost_per_million * 3.6)

    # Binary search over concurrency [1, 1024]
    lo, hi = 1, 1024
    found = False
    best_conc = None
    best_tok_s = 0.0

    # First check if we can ever meet the target
    max_tok_s = throughput_at_concurrency_fn(hi)
    if max_tok_s < required_tok_s:
        return {
            "found": False,
            "breakeven_concurrency": None,
            "required_tok_s": round(required_tok_s, 2),
            "achieved_tok_s": round(max_tok_s, 2),
            "actual_cost_per_million": round(
                hourly_cost / (max_tok_s * 3.6), 4
            ) if max_tok_s > 0 else float("inf"),
        }

    # Binary search for minimum concurrency
    while lo <= hi:
        mid = (lo + hi) // 2
        tok_s = throughput_at_concurrency_fn(mid)

        if tok_s >= required_tok_s:
            found = True
            best_conc = mid
            best_tok_s = tok_s
            hi = mid - 1
        else:
            lo = mid + 1

    if found and best_tok_s > 0:
        actual_cost = hourly_cost / (best_tok_s * 3.6)
    else:
        actual_cost = float("inf")

    return {
        "found": found,
        "breakeven_concurrency": best_conc,
        "required_tok_s": round(required_tok_s, 2),
        "achieved_tok_s": round(best_tok_s, 2),
        "actual_cost_per_million": round(actual_cost, 4),
    }


def analyze_cost(hw_name, instance_type, num_gpus, pricing_dict,
                 concurrency_throughputs):
    """Full cost analysis across concurrency levels.

    Calculates cost efficiency metrics for each concurrency level
    and identifies the optimal operating point.

    Args:
        hw_name: Hardware configuration name (for labeling).
        instance_type: EC2 instance type (key into pricing_dict).
        num_gpus: Number of GPUs in the serving configuration.
        pricing_dict: Dict mapping instance_type -> hourly_cost_usd.
        concurrency_throughputs: List of (concurrency, total_tok_s) tuples.

    Returns:
        Dict with:
        - hw_name: Hardware name.
        - instance_type: EC2 instance type.
        - num_gpus: Number of GPUs.
        - hourly_cost: Hourly cost in USD.
        - concurrency_results: List of per-concurrency cost dicts.
        - best_cost_per_million: Lowest cost per million tokens.
        - best_concurrency: Concurrency at lowest cost.
        - best_tokens_per_dollar: Highest tokens per dollar.
    """
    hourly_cost = pricing_dict.get(instance_type, 0.0)

    if hourly_cost <= 0:
        return {
            "hw_name": hw_name,
            "instance_type": instance_type,
            "num_gpus": num_gpus,
            "hourly_cost": 0.0,
            "concurrency_results": [],
            "best_cost_per_million": float("inf"),
            "best_concurrency": None,
            "best_tokens_per_dollar": 0.0,
        }

    concurrency_results = []
    best_cost = float("inf")
    best_conc = None
    best_tpd = 0.0

    for conc, tok_s in concurrency_throughputs:
        cost_info = calculate_cost_efficiency(tok_s, num_gpus, hourly_cost)

        entry = {
            "concurrency": conc,
            "total_tok_s": tok_s,
            "tokens_per_dollar": cost_info["tokens_per_dollar"],
            "cost_per_million_tokens": cost_info["cost_per_million_tokens"],
            "tokens_per_sec_per_gpu": cost_info["tokens_per_sec_per_gpu"],
        }
        concurrency_results.append(entry)

        # Track best cost efficiency
        cpm = cost_info["cost_per_million_tokens"]
        if cpm < best_cost and tok_s > 0:
            best_cost = cpm
            best_conc = conc
            best_tpd = cost_info["tokens_per_dollar"]

    return {
        "hw_name": hw_name,
        "instance_type": instance_type,
        "num_gpus": num_gpus,
        "hourly_cost": hourly_cost,
        "concurrency_results": concurrency_results,
        "best_cost_per_million": round(best_cost, 4) if best_cost < float("inf") else float("inf"),
        "best_concurrency": best_conc,
        "best_tokens_per_dollar": round(best_tpd, 2),
    }


def generate_cost_recommendations(all_results):
    """Generate cost recommendations.

    Args:
        all_results: List of dicts from analyze_cost (one per hardware).

    Returns:
        List of recommendation strings.
    """
    recs = []

    if not all_results:
        return ["No cost analysis results available."]

    # Find best hardware for cost efficiency
    valid = [r for r in all_results if r["best_cost_per_million"] < float("inf")]

    if not valid:
        recs.append(
            "No valid cost data — check that throughput and pricing are configured."
        )
        return recs

    # Sort by cost per million tokens (ascending = cheapest first)
    ranked = sorted(valid, key=lambda r: r["best_cost_per_million"])

    best = ranked[0]
    recs.append(
        f"Best cost efficiency: {best['hw_name']} ({best['instance_type']}) "
        f"at ${best['best_cost_per_million']:.2f}/M tokens "
        f"(concurrency={best['best_concurrency']}, "
        f"{best['best_tokens_per_dollar']:.0f} tokens/$)."
    )

    if len(ranked) > 1:
        worst = ranked[-1]
        ratio = worst["best_cost_per_million"] / best["best_cost_per_million"]
        recs.append(
            f"Cost range: {best['hw_name']} is {ratio:.1f}x cheaper than "
            f"{worst['hw_name']} (${best['best_cost_per_million']:.2f} vs "
            f"${worst['best_cost_per_million']:.2f} per M tokens)."
        )

    # Check for under-utilized expensive instances
    for r in all_results:
        if r["hourly_cost"] <= 0:
            continue
        results = r.get("concurrency_results", [])
        if not results:
            continue

        # Low concurrency cost check
        low_conc = [cr for cr in results if cr["concurrency"] <= 4]
        high_conc = [cr for cr in results if cr["concurrency"] >= 64]

        if low_conc and high_conc:
            low_cost = low_conc[0]["cost_per_million_tokens"]
            high_cost = high_conc[-1]["cost_per_million_tokens"]
            if low_cost > 3 * high_cost and low_cost < float("inf"):
                recs.append(
                    f"Concurrency matters: {r['hw_name']} at low concurrency "
                    f"costs ${low_cost:.2f}/M tokens vs ${high_cost:.2f}/M at "
                    f"high concurrency — {low_cost / high_cost:.0f}x improvement. "
                    f"Maximize batch utilization for cost efficiency."
                )

    # Compare per-GPU efficiency across hardware
    gpu_efficiencies = []
    for r in all_results:
        results = r.get("concurrency_results", [])
        if results:
            best_tpg = max(cr["tokens_per_sec_per_gpu"] for cr in results)
            gpu_efficiencies.append((r["hw_name"], best_tpg, r["hourly_cost"]))

    if len(gpu_efficiencies) > 1:
        gpu_efficiencies.sort(key=lambda x: x[1], reverse=True)
        top = gpu_efficiencies[0]
        bottom = gpu_efficiencies[-1]
        if bottom[1] > 0:
            recs.append(
                f"Per-GPU throughput: {top[0]} achieves {top[1]:.0f} tok/s/GPU "
                f"vs {bottom[0]} at {bottom[1]:.0f} tok/s/GPU "
                f"({top[1] / bottom[1]:.1f}x)."
            )

    return recs


def export_cost_results_csv(all_results, filename):
    """Export cost analysis results to CSV.

    Args:
        all_results: List of dicts from analyze_cost.
        filename: Output CSV file path.
    """
    rows = []

    for r in all_results:
        for cr in r.get("concurrency_results", []):
            rows.append({
                "hardware": r["hw_name"],
                "instance_type": r["instance_type"],
                "num_gpus": r["num_gpus"],
                "hourly_cost": r["hourly_cost"],
                "concurrency": cr["concurrency"],
                "total_tok_s": cr["total_tok_s"],
                "tokens_per_dollar": cr["tokens_per_dollar"],
                "cost_per_million_tokens": cr["cost_per_million_tokens"],
                "tokens_per_sec_per_gpu": cr["tokens_per_sec_per_gpu"],
            })

    if rows:
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
