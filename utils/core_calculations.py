"""
Core calculation functions shared across multiple analysis phases.

Contains fundamental model memory, training time, and memory breakdown calculations
used by multiple phases of the analysis pipeline.
"""

# Import configurations
from configuration.project_config import *
from configuration.advanced_config import *


def calculate_model_memory_gb(
    variant,
    layers_per_gpu,
    experts_per_gpu,
    param_bytes,
    vocab,
    layer_norm,
    ffn_weight_matrices,
    num_embeddings_tables,
    tp=1,
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
        tp: Tensor parallelism degree (default 1). TP shards attention,
            FFN, expert, and embedding weights across tp GPUs.

    Returns:
        float: Model memory in GB
    """
    d = variant["d"]
    layers = variant["layers"]
    moe_layers = layers - variant["dense_layers"]

    # When using pipeline parallelism, only a fraction of MoE layers are on this GPU.
    # Approximate: distribute MoE layers proportionally to total layers.
    moe_layers_per_gpu = max(0, layers_per_gpu - min(variant["dense_layers"], layers_per_gpu))
    # More precise: ratio of MoE layers in the full model applied to layers_per_gpu
    if layers > 0 and moe_layers > 0:
        moe_layers_per_gpu = int(layers_per_gpu * moe_layers / layers)

    attn_mem = layers_per_gpu * variant["attn_per_layer"] * param_bytes
    routed_mem = experts_per_gpu * moe_layers_per_gpu * variant["expert_each"] * param_bytes
    shared_mem = moe_layers_per_gpu * variant["shared_each"] * param_bytes
    router_mem = moe_layers_per_gpu * variant["router_each"] * param_bytes
    ln_mem = layers_per_gpu * layer_norm * d * param_bytes
    dense_layers_per_gpu = layers_per_gpu - moe_layers_per_gpu
    dense_ffn_mem = (
        dense_layers_per_gpu
        * ffn_weight_matrices
        * d
        * variant["dense_ffn"]
        * param_bytes
    )
    embed_mem = num_embeddings_tables * vocab * d * param_bytes

    # TP shards attention, FFN, expert, and embedding weights across tp GPUs.
    # Layer norm and router weights are small and replicated (not sharded).
    model_mem_bytes = (
        (attn_mem + routed_mem + shared_mem + dense_ffn_mem + embed_mem) / tp
        + ln_mem
        + router_mem
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
    ep=1,
    pp=1,
    vp=1,
    nccl_ep_scaling_factor=0.0,
    fragmentation_factor=1.0,
    experts_per_gpu=0,
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
        nccl_mem_buf: NCCL base buffer memory in GB
        optim_bytes: Optimizer bytes per parameter
        ep: Expert parallelism degree (default 1)
        pp: Pipeline parallelism degree (default 1)
        vp: Virtual pipeline parallelism degree (default 1)
        nccl_ep_scaling_factor: Additional NCCL GB per EP rank (default 0.0)
        fragmentation_factor: PyTorch allocator fragmentation multiplier (default 1.0)
        experts_per_gpu: Number of experts per GPU (default 0)

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
    is_moe = variant.get("expert_ffn", 0) > 0
    moe_layers = variant["layers"] - variant.get("dense_layers", variant["layers"])
    total_layers = variant["layers"]
    moe_layers_per_gpu = int(layers_per_gpu * moe_layers / total_layers) if total_layers > 0 else 0

    # --- Optimizer memory with EP/DP interaction fix ---
    # When EP subdivides DP (dp > ep), MoE params are only shared within dp/ep ranks.
    # When dp <= ep, optimizer uses full dp normally (all experts already distributed).
    if is_moe and ep > 1 and dp > ep:
        eff_dp_moe = dp // ep
    else:
        eff_dp_moe = dp

    # Estimate fraction of model memory that is MoE expert params vs non-MoE
    if is_moe and model_mem_gb > 0:
        # Expert memory fraction based on architecture (must match calculate_model_memory_gb)
        routed_mem = experts_per_gpu * moe_layers_per_gpu * variant["expert_each"] * param_bytes
        total_model_bytes = model_mem_gb * 1e9
        moe_frac = min(1.0, routed_mem / total_model_bytes) if total_model_bytes > 0 else 0
        dense_frac = 1.0 - moe_frac
    else:
        moe_frac = 0.0
        dense_frac = 1.0

    # Calculate gradient and optimizer memory based on ZeRO strategy
    if zero_strategy == 1:
        grad_mem = model_mem_gb
        # Optimizer: dense params sharded across full DP, MoE params across eff_dp_moe
        optim_dense = model_mem_gb * dense_frac * optim_bytes / dp
        optim_moe = model_mem_gb * moe_frac * optim_bytes / eff_dp_moe
        optim_mem = optim_dense + optim_moe
    elif zero_strategy == 2:
        grad_mem = model_mem_gb / dp
        optim_dense = model_mem_gb * dense_frac * optim_bytes / dp
        optim_moe = model_mem_gb * moe_frac * optim_bytes / eff_dp_moe
        optim_mem = optim_dense + optim_moe
    elif zero_strategy == 3:
        grad_mem = model_mem_gb / dp
        optim_dense = model_mem_gb * dense_frac * optim_bytes / dp
        optim_moe = model_mem_gb * moe_frac * optim_bytes / eff_dp_moe
        optim_mem = optim_dense + optim_moe
    else:
        raise ValueError(f"Unknown ZeRO strategy: {zero_strategy}")

    # --- Activation memory ---
    # For MoE layers, activations include expert intermediates (higher per-layer cost)
    if is_moe and moe_layers_per_gpu > 0:
        # MoE activation per layer: attention acts + routed expert intermediates
        # Expert intermediate = tokens * topk * expert_ffn * 2 (input + output of expert FFN)
        expert_ffn = variant.get("expert_ffn", 0)
        moe_act_per_layer = seq_len * micro * (d * empirical_act_multiplier + topk * expert_ffn * 2 * param_bytes)
        dense_act_per_layer = seq_len * micro * d * empirical_act_multiplier
        dense_layers_on_gpu = layers_per_gpu - moe_layers_per_gpu
        act_total_full = (moe_layers_per_gpu * moe_act_per_layer + dense_layers_on_gpu * dense_act_per_layer)
    else:
        act_total_full = layers_per_gpu * seq_len * micro * d * empirical_act_multiplier

    activation_mem = act_total_full * selective_act_checkpointing_multiplier / 1e9

    # VP activation overhead: interleaved schedule keeps more activations in-flight
    # In interleaved 1F1B, warmup fills (PP-1)*VP micro-batches before steady state.
    # Each micro-batch occupies layers_per_virtual_chunk = layers_per_gpu / VP layers.
    # Total layer-activations stored = (PP-1) * layers_per_gpu (independent of VP).
    # But lower VP = fewer but larger chunks = less overhead from per-chunk metadata.
    if vp > 1 and pp > 1:
        layers_per_virtual_chunk = layers_per_gpu // vp
        inflight_micros = (pp - 1) * vp  # warmup micro-batches
        # In-flight activations use the same per-layer cost but without checkpointing reduction
        if is_moe and moe_layers_per_gpu > 0:
            expert_ffn = variant.get("expert_ffn", 0)
            avg_act_per_layer = seq_len * micro * (d * empirical_act_multiplier + topk * expert_ffn * 2 * param_bytes)
        else:
            avg_act_per_layer = seq_len * micro * d * empirical_act_multiplier
        vp_act_overhead = inflight_micros * layers_per_virtual_chunk * avg_act_per_layer / 1e9
        activation_mem += vp_act_overhead

    # --- Buffer memory ---
    # All-to-all routing buffers
    a2a_buf = (
        seq_len * micro * topk * d * fwd_bwd_routing_buff_passes * param_bytes / 1e9
    )

    # MoE dispatch buffers: send + recv permutation buffers per MoE layer
    moe_dispatch_buf = 0.0
    if is_moe and experts_per_gpu > 0:
        moe_dispatch_buf = (
            moe_layers_per_gpu * 2 * micro * seq_len * d * param_bytes
            * (topk / experts_per_gpu) / 1e9
        )

    # NCCL workspace: base + scaling with EP group size
    nccl_workspace = nccl_mem_buf + nccl_ep_scaling_factor * max(0, ep - 1)

    # PP pipeline send/recv buffers: each stage needs send+recv buffers for activations
    pp_buf = 0.0
    if pp > 1:
        # Two buffers (send + recv) each holding one micro-batch's activation slice
        pp_buf = 2 * micro * seq_len * d * param_bytes / 1e9

    buffer_mem = a2a_buf + moe_dispatch_buf + nccl_workspace + pp_buf

    # --- Total with fragmentation ---
    raw_total = model_mem_gb + grad_mem + optim_mem + activation_mem + buffer_mem
    total_mem = raw_total * fragmentation_factor

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
    tp=1,
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
        tp: Tensor parallelism degree (default 1)

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
        tp=tp,
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
            ep=EP,
            pp=PP,
            vp=VP,
            nccl_ep_scaling_factor=NCCL_EP_SCALING_FACTOR,
            fragmentation_factor=FRAGMENTATION_FACTOR,
            experts_per_gpu=experts_per_gpu,
        )

        # Check if this micro batch size fits in memory
        if memory_breakdown["total"] <= gpu_mem_gb * threshold:
            viable_micros.append(micro)

    return sorted(viable_micros)
