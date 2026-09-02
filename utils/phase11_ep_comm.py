"""
Phase 11: Expert Parallel Communication Analysis for MoE Inference.

Analyzes EP communication overhead for Mixture-of-Experts models during
inference. With EP > 1, experts are distributed across GPUs, requiring
all-to-all communication for dispatch (tokens -> experts) and combine
(expert outputs -> tokens).

Key formulas:
  # Per-token dispatch volume (sending token to topk experts across EP GPUs):
  dispatch_per_token = hidden_dim * dtype_bytes * topk * (ep - 1) / ep

  # Per-token combine volume (gathering expert outputs back):
  combine_per_token = hidden_dim * dtype_bytes * topk * (ep - 1) / ep

  # Total EP traffic per decode step (batch tokens):
  ep_traffic_per_step = (dispatch_per_token + combine_per_token) * batch_size * moe_layers

  # Where moe_layers = total_layers - dense_layers

  # EP latency per step:
  ep_latency = ep_traffic_per_step / interconnect_bw

Notes:
  - Skip analysis entirely for dense models (expert_ffn == 0).
  - MoE layers = layers - dense_layers.
  - TopK from project_config.TOPK.
  - Uses intra-node BW if EP fits in 1 node, else inter-node.
"""

import math
import csv

from configuration.project_config import TOPK


def _get_moe_layers(variant):
    """Get the number of MoE layers in the model.

    Args:
        variant: Model variant dict (has layers, dense_layers).

    Returns:
        Number of MoE layers (0 for dense models).
    """
    total_layers = variant.get("layers", 0)
    dense_layers = variant.get("dense_layers", total_layers)
    return total_layers - dense_layers


def _is_moe_model(variant):
    """Check if the variant is a MoE model.

    Args:
        variant: Model variant dict.

    Returns:
        True if model has MoE layers, False for dense models.
    """
    expert_ffn = variant.get("expert_ffn", 0)
    moe_layers = _get_moe_layers(variant)
    return expert_ffn > 0 and moe_layers > 0


def calculate_ep_dispatch_volume(variant, ep, batch_size, dtype_bytes):
    """Dispatch + combine bytes per decode step for MoE models.

    Each token is routed to topk experts. With EP > 1, a fraction
    (ep-1)/ep of those expert activations must be sent to remote GPUs
    (dispatch) and the results gathered back (combine).

    Formulas:
        dispatch_per_token = hidden_dim * dtype_bytes * topk * (ep - 1) / ep
        combine_per_token  = hidden_dim * dtype_bytes * topk * (ep - 1) / ep
        total_per_token = dispatch_per_token + combine_per_token
        total_per_step = total_per_token * batch_size * moe_layers

    Args:
        variant: Model variant dict (has d, layers, dense_layers, expert_ffn).
        ep: Expert parallelism degree.
        batch_size: Number of tokens per decode step.
        dtype_bytes: Bytes per activation element (2 for BF16, 1 for FP8).

    Returns:
        Dict with:
        - dispatch_per_token_bytes: Dispatch bytes per token per MoE layer.
        - combine_per_token_bytes: Combine bytes per token per MoE layer.
        - total_per_token_bytes: Total bytes per token per MoE layer.
        - total_per_step_bytes: Total bytes per decode step.
        - moe_layers: Number of MoE layers.
        - topk: TopK experts used.
        - is_moe: True if MoE model.
    """
    if not _is_moe_model(variant):
        return {
            "dispatch_per_token_bytes": 0,
            "combine_per_token_bytes": 0,
            "total_per_token_bytes": 0,
            "total_per_step_bytes": 0,
            "moe_layers": 0,
            "topk": TOPK,
            "is_moe": False,
        }

    if ep <= 1:
        return {
            "dispatch_per_token_bytes": 0,
            "combine_per_token_bytes": 0,
            "total_per_token_bytes": 0,
            "total_per_step_bytes": 0,
            "moe_layers": _get_moe_layers(variant),
            "topk": TOPK,
            "is_moe": True,
        }

    hidden_dim = variant["d"]
    moe_layers = _get_moe_layers(variant)
    topk = TOPK

    # Per-token dispatch: send activation to topk remote experts
    # Fraction (ep-1)/ep goes to remote GPUs
    dispatch_per_token = hidden_dim * dtype_bytes * topk * (ep - 1) / ep

    # Per-token combine: gather expert outputs back (symmetric)
    combine_per_token = hidden_dim * dtype_bytes * topk * (ep - 1) / ep

    total_per_token = dispatch_per_token + combine_per_token

    # Total per decode step: across all tokens and MoE layers
    total_per_step = total_per_token * batch_size * moe_layers

    return {
        "dispatch_per_token_bytes": dispatch_per_token,
        "combine_per_token_bytes": combine_per_token,
        "total_per_token_bytes": total_per_token,
        "total_per_step_bytes": total_per_step,
        "moe_layers": moe_layers,
        "topk": topk,
        "is_moe": True,
    }


def estimate_ep_latency_ms(variant, hw, ep, batch_size, dtype_bytes):
    """EP all-to-all latency per decode step in milliseconds.

    Uses intra-node bandwidth if EP fits within a single node
    (ep <= gpus_per_node), otherwise uses inter-node bandwidth.

    Formula:
        ep_latency_ms = total_per_step_bytes / (interconnect_bw * 1e9) * 1000

    Args:
        variant: Model variant dict.
        hw: Hardware config dict (has intra_node_bw_gbps, inter_node_bw_gb,
            gpus_per_node).
        ep: Expert parallelism degree.
        batch_size: Number of tokens per decode step.
        dtype_bytes: Bytes per activation element.

    Returns:
        Dict with:
        - ep_latency_ms: All-to-all latency in milliseconds.
        - interconnect_bw_gbps: Bandwidth used.
        - interconnect_type: "intra-node" or "inter-node".
        - total_bytes: Total bytes communicated.
        - is_moe: True if MoE model.
    """
    if not _is_moe_model(variant) or ep <= 1:
        return {
            "ep_latency_ms": 0.0,
            "interconnect_bw_gbps": 0.0,
            "interconnect_type": "none",
            "total_bytes": 0,
            "is_moe": _is_moe_model(variant),
        }

    vol = calculate_ep_dispatch_volume(variant, ep, batch_size, dtype_bytes)
    total_bytes = vol["total_per_step_bytes"]

    gpus_per_node = hw.get("gpus_per_node", 8)
    is_intra_node = ep <= gpus_per_node

    if is_intra_node:
        bw_gbps = hw.get("intra_node_bw_gbps", 900)
        ic_type = "intra-node"
    else:
        bw_gbps = hw.get("inter_node_bw_gb", 400)
        ic_type = "inter-node"

    if bw_gbps > 0 and total_bytes > 0:
        latency_s = total_bytes / (bw_gbps * 1e9)
        latency_ms = latency_s * 1000.0
    else:
        latency_ms = 0.0

    return {
        "ep_latency_ms": round(latency_ms, 4),
        "interconnect_bw_gbps": bw_gbps,
        "interconnect_type": ic_type,
        "total_bytes": total_bytes,
        "is_moe": True,
    }


def analyze_ep_comm(variant, hw_list, ep_values, batch_sizes, dtype_bytes):
    """Sweep EP and batch sizes across hardware configurations.

    Skips analysis entirely if the model is not MoE (expert_ffn == 0).

    Args:
        variant: Model variant dict.
        hw_list: List of hardware config dicts.
        ep_values: List of EP degrees to sweep.
        batch_sizes: List of batch sizes to sweep.
        dtype_bytes: Bytes per activation element.

    Returns:
        Dict with:
        - variant_name: Model name.
        - is_moe: Whether model is MoE.
        - skipped: True if analysis was skipped (dense model).
        - results: Dict mapping hw_name -> list of per-(ep, batch) dicts.
    """
    if not _is_moe_model(variant):
        return {
            "variant_name": variant["name"],
            "is_moe": False,
            "skipped": True,
            "results": {},
        }

    results = {}

    for hw in hw_list:
        hw_name = hw["name"]
        hw_results = []

        for ep in ep_values:
            for bs in batch_sizes:
                vol = calculate_ep_dispatch_volume(variant, ep, bs, dtype_bytes)
                lat = estimate_ep_latency_ms(variant, hw, ep, bs, dtype_bytes)

                hw_results.append({
                    "ep": ep,
                    "batch_size": bs,
                    "dispatch_per_token_bytes": vol["dispatch_per_token_bytes"],
                    "combine_per_token_bytes": vol["combine_per_token_bytes"],
                    "total_per_step_bytes": vol["total_per_step_bytes"],
                    "moe_layers": vol["moe_layers"],
                    "topk": vol["topk"],
                    "ep_latency_ms": lat["ep_latency_ms"],
                    "interconnect_type": lat["interconnect_type"],
                    "interconnect_bw_gbps": lat["interconnect_bw_gbps"],
                })

        results[hw_name] = hw_results

    return {
        "variant_name": variant["name"],
        "is_moe": True,
        "skipped": False,
        "results": results,
    }


def generate_ep_recommendations(all_results):
    """Generate recommendations for EP sizing.

    Args:
        all_results: Dict from analyze_ep_comm.

    Returns:
        List of recommendation strings.
    """
    recs = []
    variant_name = all_results.get("variant_name", "Unknown")

    if all_results.get("skipped", False):
        recs.append(
            f"Skipped: {variant_name} is a dense model — EP communication "
            f"analysis not applicable."
        )
        return recs

    for hw_name, hw_results in all_results.get("results", {}).items():
        if not hw_results:
            continue

        # Separate EP=1 (no comm) from EP>1
        ep_gt1 = [r for r in hw_results if r["ep"] > 1]
        if not ep_gt1:
            continue

        # Find worst-case latency
        max_lat = max(ep_gt1, key=lambda x: x["ep_latency_ms"])
        min_lat = min(ep_gt1, key=lambda x: x["ep_latency_ms"])

        # Flag high EP latency (>5ms per decode step)
        if max_lat["ep_latency_ms"] > 5.0:
            recs.append(
                f"High EP latency: {variant_name} on {hw_name} with "
                f"EP={max_lat['ep']}, batch={max_lat['batch_size']} has "
                f"EP latency={max_lat['ep_latency_ms']:.1f} ms per decode "
                f"step. Consider reducing EP or batch size."
            )

        # Flag inter-node EP
        inter_node = [r for r in ep_gt1 if r["interconnect_type"] == "inter-node"]
        if inter_node:
            recs.append(
                f"Inter-node EP: {variant_name} on {hw_name} with "
                f"EP={inter_node[0]['ep']} requires inter-node all-to-all. "
                f"Latency is significantly higher than intra-node. "
                f"Consider keeping EP <= gpus_per_node."
            )

        # Good latency
        if max_lat["ep_latency_ms"] < 1.0:
            recs.append(
                f"Good EP overhead: {variant_name} on {hw_name} has "
                f"EP latency < 1 ms across all configs — minimal overhead."
            )

        # Batch size impact
        batch_latencies = {}
        for r in ep_gt1:
            bs = r["batch_size"]
            if bs not in batch_latencies:
                batch_latencies[bs] = []
            batch_latencies[bs].append(r["ep_latency_ms"])

        if len(batch_latencies) > 1:
            avg_by_batch = {
                bs: sum(lats) / len(lats)
                for bs, lats in batch_latencies.items()
            }
            min_bs = min(avg_by_batch, key=avg_by_batch.get)
            max_bs = max(avg_by_batch, key=avg_by_batch.get)
            if avg_by_batch[max_bs] > 2 * avg_by_batch[min_bs]:
                recs.append(
                    f"Batch impact: {variant_name} on {hw_name} EP latency "
                    f"scales with batch size (batch={max_bs}: "
                    f"{avg_by_batch[max_bs]:.1f} ms vs batch={min_bs}: "
                    f"{avg_by_batch[min_bs]:.1f} ms)."
                )

    return recs


def export_ep_results_csv(all_results, filename):
    """Export EP communication analysis results to CSV.

    Args:
        all_results: Dict from analyze_ep_comm.
        filename: Output CSV file path.
    """
    if all_results.get("skipped", False):
        return

    rows = []
    variant_name = all_results.get("variant_name", "Unknown")

    for hw_name, hw_results in all_results.get("results", {}).items():
        for r in hw_results:
            rows.append({
                "variant": variant_name,
                "hardware": hw_name,
                "ep": r["ep"],
                "batch_size": r["batch_size"],
                "moe_layers": r["moe_layers"],
                "topk": r["topk"],
                "dispatch_per_token_bytes": r["dispatch_per_token_bytes"],
                "combine_per_token_bytes": r["combine_per_token_bytes"],
                "total_per_step_bytes": r["total_per_step_bytes"],
                "ep_latency_ms": r["ep_latency_ms"],
                "interconnect_type": r["interconnect_type"],
                "interconnect_bw_gbps": r["interconnect_bw_gbps"],
            })

    if rows:
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
