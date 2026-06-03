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

    # Pipeline bubble percentage
    # Interleaved: bubble = (pp-1) / (num_microbatches * vp)
    # Standard: bubble = (pp-1) / num_microbatches
    effective_microbatches = num_microbatches * vp if vp > 1 else num_microbatches
    pipeline_bubble_pct = ((pp - 1) / effective_microbatches * 100
                           if effective_microbatches > 0 else 100)

    # Determine communication type
    # PP stages within same node use NVLink; cross-node uses EFA
    is_inter_node = pp > gpus_per_node
    comm_type = "inter-node (EFA)" if is_inter_node else "intra-node (NVLink)"

    # Time per send: inter-node P2P only uses 1-2 of 16 NICs on EFA
    # Intra-node uses NVLink at much higher effective bandwidth
    if is_inter_node:
        effective_bw_gbps = inter_node_bw_gb * EFA_PP_EFFICIENCY  # ~3.2 GB/s on EFA
    else:
        effective_bw_gbps = intra_node_bw_gbps  # e.g., 1800 GB/s NVLink
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
