"""
Phase 6: Pipeline Parallelism SendRecv Communication Analysis.

Models PP point-to-point communication overhead including activation transfers
between pipeline stages, bubble time, and EFA utilization.
"""

from configuration.project_config import *
from configuration.advanced_config import *


# PP SendRecv over EFA uses only 1-2 of 16 NICs (point-to-point)
# Empirical from nsys profiling: ~3.2 GB/s achieved vs 200 GB/s theoretical
EFA_PP_EFFICIENCY = 0.016


def calculate_pp_communication(layers, hidden_dim, seq_len, mbs, gbs, pp, vp, dp, ep,
                                dtype_bytes=2, inter_node_bw_gb=400,
                                intra_node_bw_gbps=1800, gpus_per_node=8):
    """
    Compute PP SendRecv traffic, call counts, and timing estimates.

    Args:
        layers: Total model layers
        hidden_dim: Model hidden dimension
        seq_len: Sequence length
        mbs: Micro batch size
        gbs: Global batch size (sequences)
        pp: Pipeline parallelism degree
        vp: Virtual pipeline parallelism degree
        dp: Data parallelism degree
        ep: Expert parallelism degree (reserved for future EP-aware stage mapping)
        dtype_bytes: Bytes per activation element (2 for BF16)
        inter_node_bw_gb: Inter-node bandwidth in GB/s (EFA)
        intra_node_bw_gbps: Intra-node bandwidth in GB/s (NVLink)
        gpus_per_node: GPUs per node

    Returns:
        dict with PP communication metrics
    """
    if pp <= 1:
        return {
            "activation_size_mb": 0,
            "sends_per_microbatch": 0,
            "num_microbatches": gbs // (mbs * dp) if (mbs * dp) > 0 else 0,
            "total_sends_per_step": 0,
            "total_traffic_gb": 0,
            "pipeline_bubble_pct": 0,
            "time_per_send_us": 0,
            "estimated_pp_time_ms": 0,
            "comm_type": "none (PP=1)",
            "efa_utilization_pct": 0,
        }

    # Activation size per P2P send: one micro-batch's hidden states
    activation_size_bytes = mbs * seq_len * hidden_dim * dtype_bytes
    activation_size_mb = activation_size_bytes / 1e6

    # Number of sends per micro-batch:
    # Interleaved (VP>1): 2*(pp*vp - 1) sends across all virtual chunks (fwd+bwd)
    # Standard (VP=1): 2*(pp-1) sends (fwd+bwd)
    if vp > 1:
        sends_per_microbatch = 2 * (pp * vp - 1)
    else:
        sends_per_microbatch = 2 * (pp - 1)

    # Number of micro-batches per training step
    num_microbatches = gbs // (mbs * dp) if (mbs * dp) > 0 else 1

    # Total sends per step
    total_sends_per_step = sends_per_microbatch * num_microbatches

    # Total traffic
    total_traffic_gb = total_sends_per_step * activation_size_bytes / 1e9

    # Pipeline bubble fraction.
    #
    # Standard GPipe / 1F1B:  (pp - 1) / (m + pp - 1)
    # Interleaved (VP = v):   (pp - 1) / (v*m + pp - 1)
    #
    # The denominator MUST include the (pp - 1) fill/drain term. The previous form
    # was (pp - 1) / m, which omits it and is only a large-m approximation -- for
    # m < pp it returns a bubble ABOVE 100%, which is impossible for a fraction of
    # idle time. Measured: pp=16, m=8 gave 187.5%, and this value feeds the Phase 3
    # time estimate directly as a multiplier.
    #
    # Interleaving divides the bubble by the virtual-stage count v, which is the
    # point of the interleaved schedule.
    effective_microbatches = num_microbatches * vp if vp > 1 else num_microbatches
    denom = effective_microbatches + pp - 1
    pipeline_bubble_pct = ((pp - 1) / denom * 100) if denom > 0 else 100.0

    # Determine communication type
    # PP stages within same node use NVLink; cross-node uses EFA
    is_inter_node = pp > gpus_per_node
    comm_type = "inter-node (EFA)" if is_inter_node else "intra-node (NVLink)"

    # Time per send. A PP point-to-point send engages only a small fraction of the
    # node's EFA capacity, because it is one rank talking to one rank rather than a
    # collective spread across all NICs.
    #
    # EFA_PP_EFFICIENCY is EMPIRICAL (0.016). Its two previously stated rationales
    # were both wrong and are not restated here: "1-2 of 16 NICs" would be 0.0625
    # or 0.125, and 400 GB/s * 0.016 is 6.4 GB/s, not the "~3.2 GB/s" claimed.
    # Treat it as a calibration constant, not a derived one.
    if is_inter_node:
        effective_bw_gbps = inter_node_bw_gb * EFA_PP_EFFICIENCY
    else:
        effective_bw_gbps = intra_node_bw_gbps  # full NVLink, e.g. 1800 GB/s
    time_per_send_s = (activation_size_bytes / 1e9) / effective_bw_gbps if effective_bw_gbps > 0 else 0
    time_per_send_us = time_per_send_s * 1e6

    # Estimated total PP communication time per step
    # Not all sends are serialized - pipeline has overlap. Exposed time ≈ bubble sends.
    # During steady state, sends overlap with compute. Exposed = warmup + cooldown sends.
    exposed_sends = 2 * (pp - 1)  # warmup fwd + cooldown bwd (serialized)
    estimated_pp_time_ms = exposed_sends * time_per_send_us / 1000

    # EFA utilization: fraction of peak bandwidth actually used
    efa_utilization_pct = EFA_PP_EFFICIENCY * 100

    return {
        "activation_size_mb": activation_size_mb,
        "sends_per_microbatch": sends_per_microbatch,
        "num_microbatches": num_microbatches,
        "total_sends_per_step": total_sends_per_step,
        "total_traffic_gb": total_traffic_gb,
        "pipeline_bubble_pct": pipeline_bubble_pct,
        "time_per_send_us": time_per_send_us,
        "estimated_pp_time_ms": estimated_pp_time_ms,
        "comm_type": comm_type,
        "efa_utilization_pct": efa_utilization_pct,
    }
