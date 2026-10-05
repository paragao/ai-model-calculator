"""
Phase 7: Tensor-parallel communication.

Tensor parallelism was previously INVISIBLE to this calculator in both
directions: activation memory was identical at tp = 1, 2, 4 and 8, and no
communication model accepted `tp` at all. A repo-wide grep for `all_reduce` /
`allreduce` / `tp_comm` returned one hit, and it was prose. TP therefore had
neither a modelled cost nor a modelled benefit, so any search over parallelism
would have chosen the largest TP available because it was free.

The memory half is fixed in core_calculations via ACT_TP_REPLICATED_FRACTION.
This module is the cost half.

FIRST PRINCIPLES
----------------
Megatron-LM partitions each transformer block so that exactly two points need a
collective in the forward pass and two in the backward pass:

  forward   g: all-reduce after the ROW-parallel self-attention output projection
            g: all-reduce after the ROW-parallel MLP second linear
  backward  f: all-reduce of the gradient entering the COLUMN-parallel attention
            f: all-reduce of the gradient entering the COLUMN-parallel MLP

So FOUR all-reduces per layer per micro-batch, each over the full block
activation tensor of shape (seq_len, micro, hidden) -- the tensor is NOT
partitioned at those points, which is the whole reason a collective is needed.

A ring all-reduce of V bytes across t ranks moves 2*V*(t-1)/t bytes PER RANK
(reduce-scatter leg + all-gather leg). At t=1 there is no collective at all, so
the volume is exactly zero and this module contributes nothing -- which makes it
inert at the shipped TP=1 by construction, not by a special case.

TOPOLOGY
--------
TP is the INNERMOST dimension in Megatron's default `tp-cp-ep-dp-pp` rank order,
so a TP group is contiguous and lands inside one node whenever tp <= the node's
GPU count. That is the case for every TP a real recipe uses (2, 4 or 8 on an
8-GPU node), so the default link is NVLink. tp > gpus_per_node is modelled as
inter-node, and flagged, because it is a configuration to avoid rather than one
to cost cheaply -- the opposite error from phase 6's PP link test, which asked
`pp > gpus_per_node` and so called an outermost-dimension PP group intra-node.

WHY THIS IS NOT ADDED TO TRAINING TIME
--------------------------------------
MFU in this calculator is an END-TO-END figure: a measured MFU already contains
the cycles a GPU spends idle waiting on collectives, pipeline bubbles and every
other stall, because it is achieved-FLOPs over peak-FLOPs across the whole run.
Adding a modelled communication time on top of an end-to-end MFU charges the
same stall twice. So this module's output is a DIAGNOSTIC: it attributes where
the MFU shortfall comes from and lets a parallelism search prefer configurations
with less exposed communication, without re-deriving a wall-clock time that the
measured MFU already determines.
"""

# Import configurations
from configuration.project_config import *
from configuration.advanced_config import *


def calculate_tp_communication(
    layers_per_gpu,
    hidden_dim,
    seq_len,
    micro,
    tp,
    num_microbatches=1,
    dtype_bytes=2,
    intra_node_bw_gbps=900.0,
    inter_node_bw_gb=400.0,
    gpus_per_node=8,
):
    """TP all-reduce volume and time per training step, per GPU.

    Args:
        layers_per_gpu: Transformer layers resident on this GPU (layers // pp).
        hidden_dim: Model hidden dimension `d`.
        seq_len: Sequence length PER RANK -- pass seq_len/cp when context
            parallelism is in use, because CP shards the sequence and the
            all-reduced tensor is (seq_len/cp, micro, hidden). Phase 6 and the
            MoE all-to-all model both omit this division, which overstates their
            volumes cp-fold; this one takes it from the caller deliberately.
        micro: Micro-batch size (sequences per GPU).
        tp: Tensor parallelism degree. At tp<=1 every return is 0.
        num_microbatches: Micro-batches per step (grad accumulation count).
        dtype_bytes: Bytes per activation element on the wire (2 for BF16).
        intra_node_bw_gbps: Intra-node link bandwidth, GB/s.
        inter_node_bw_gb: Inter-node link bandwidth, GB/s.
        gpus_per_node: GPUs per node, used only to pick the link.

    Returns:
        dict with per-layer, per-micro-batch and per-step volumes in GB, the
        selected link, the per-step time in ms, and `is_inter_node` so a caller
        can flag a TP group that spans nodes.
    """
    if not tp or tp <= 1:
        return {
            "tp": 1,
            "collectives_per_layer": 0,
            "volume_per_layer_gb": 0.0,
            "volume_per_microbatch_gb": 0.0,
            "volume_per_step_gb": 0.0,
            "comm_type": "none (tp=1)",
            "is_inter_node": False,
            "effective_bw_gbps": 0.0,
            "time_per_step_ms": 0.0,
        }

    # The tensor each collective reduces: the full block activation.
    tensor_bytes = seq_len * micro * hidden_dim * dtype_bytes

    # Ring all-reduce moves 2V(t-1)/t per rank.
    one_all_reduce_bytes = 2.0 * tensor_bytes * (tp - 1) / tp

    # Four per layer per micro-batch: two forward (g), two backward (f).
    collectives_per_layer = 4
    per_layer_bytes = collectives_per_layer * one_all_reduce_bytes
    per_microbatch_bytes = per_layer_bytes * layers_per_gpu
    per_step_bytes = per_microbatch_bytes * max(1, num_microbatches)

    # TP is innermost, so a group is contiguous and intra-node for tp <= node size.
    is_inter_node = tp > gpus_per_node
    effective_bw = inter_node_bw_gb if is_inter_node else intra_node_bw_gbps
    comm_type = "inter-node (EFA)" if is_inter_node else "intra-node (NVLink)"

    time_per_step_ms = (
        (per_step_bytes / 1e9) / effective_bw * 1000.0 if effective_bw > 0 else 0.0
    )

    return {
        "tp": tp,
        "collectives_per_layer": collectives_per_layer,
        "volume_per_layer_gb": per_layer_bytes / 1e9,
        "volume_per_microbatch_gb": per_microbatch_bytes / 1e9,
        "volume_per_step_gb": per_step_bytes / 1e9,
        "comm_type": comm_type,
        "is_inter_node": is_inter_node,
        "effective_bw_gbps": effective_bw,
        "time_per_step_ms": time_per_step_ms,
    }
