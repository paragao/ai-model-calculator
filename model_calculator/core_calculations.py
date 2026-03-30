"""
Core calculation functions shared across multiple analysis phases.

Contains fundamental model memory, training time, and memory breakdown calculations
used by multiple phases of the analysis pipeline.
"""

# Import configurations
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
