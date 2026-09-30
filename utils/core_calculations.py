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
    ep=None,
):
    """
    Calculate model memory in GB for a given variant.

    Args:
        variant: Variant dictionary with architecture specs
        layers_per_gpu: Number of layers per GPU (total_layers // PP)
        experts_per_gpu: Number of experts per GPU (N_EXPERTS // EP)
        param_bytes: Bytes per parameter (2 for BF16, 1 for FP8)
        vocab: Vocabulary size. Used only as a FALLBACK -- a variant carrying
            its own "vocab" key overrides it, because the global default is
            correct for just the Qwen3 family (Llama-3.x 128256,
            Llama-4 202048, Qwen2.5 152064, Qwen3.5 248320, DeepSeek 129280).
        layer_norm: Layer norm constant (typically 2)
        ffn_weight_matrices: FFN weight matrices constant (typically 3)
        num_embeddings_tables: Number of embedding tables (typically 2:
            input embedding + output LM head). Reduced to 1 for a variant
            with "tied_embeddings": True, where the LM head IS the input
            embedding matrix and counting it twice overstates the model.
        tp: Tensor parallelism degree (default 1). TP shards attention,
            FFN, expert, and embedding weights across tp GPUs.
        ep: Expert parallelism degree used to re-derive experts_per_gpu from a
            variant's own "num_experts". Defaults to the global EP. Pass it
            explicitly to drive this function at an EP other than the configured
            one -- notably ep=1, which places every expert on one rank and so
            returns the FULL model size, the form that can be reconciled against
            a published parameter count. Without this the global EP was applied
            unconditionally and the experts_per_gpu argument was silently
            discarded for any variant declaring num_experts.

    Returns:
        float: Model memory in GB
    """
    d = variant["d"]
    layers = variant["layers"]
    moe_layers = layers - variant["dense_layers"]

    # experts_per_gpu arrives as the GLOBAL N_EXPERTS // EP, which is right only
    # for models that actually have N_EXPERTS experts. A variant declaring its
    # own count overrides it, because the global is wrong in both directions:
    # Llama-4-Scout has 16 experts (global 128 overstates routed memory 8x),
    # while Qwen3.5-397B has 512 and DeepSeek 256 (understated 4x and 2x).
    # Note moe_layer_params in the catalog already reflects the true count, but
    # it is metadata -- the memory path below uses expert_each * experts_per_gpu.
    variant_experts = variant.get("num_experts")
    effective_ep = EP if ep is None else ep
    if variant_experts and effective_ep > 0:
        experts_per_gpu = max(1, variant_experts // effective_ep)

    # When using pipeline parallelism, only a fraction of MoE layers are on this GPU.
    # Approximate: distribute MoE layers proportionally to total layers.
    moe_layers_per_gpu = max(0, layers_per_gpu - min(variant["dense_layers"], layers_per_gpu))
    # More precise: ratio of MoE layers in the full model applied to layers_per_gpu
    if layers > 0 and moe_layers > 0:
        moe_layers_per_gpu = int(layers_per_gpu * moe_layers / layers)

    # Attention memory, split by LAYER TYPE for a hybrid-attention model.
    #
    # A hybrid family such as Qwen3.5 repeats [linear, linear, linear, full] --
    # full_attention_interval=4 -- and a Gated DeltaNet "linear_attention" layer
    # has a completely different weight set from a gated full-attention layer. It
    # is NOT smaller: measured against the published configs, a GDN layer is
    # 1.10x to 1.24x a full-attention layer, so charging every layer at
    # attn_per_layer UNDERSTATES a hybrid model's attention memory.
    #
    # A variant carrying "linear_attn_per_layer" is hybrid and is split here.
    # Without it every layer is full attention, which is right for all the
    # uniform-attention families.
    linear_per_layer = variant.get("linear_attn_per_layer")
    interval = variant.get("full_attention_interval", 1) or 1
    if linear_per_layer and interval > 1:
        # Apportion this GPU's layers over the repeating pattern, the same way
        # moe_layers_per_gpu is apportioned above.
        full_attn_layers_per_gpu = layers_per_gpu // interval
        linear_attn_layers_per_gpu = layers_per_gpu - full_attn_layers_per_gpu
        attn_mem = (
            full_attn_layers_per_gpu * variant["attn_per_layer"]
            + linear_attn_layers_per_gpu * linear_per_layer
        ) * param_bytes
    else:
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
    embed_tables = 1 if variant.get("tied_embeddings") else num_embeddings_tables
    embed_mem = embed_tables * variant.get("vocab", vocab) * d * param_bytes

    # TP shards attention, FFN, expert, and embedding weights across tp GPUs.
    # Layer norm and router weights are small and replicated (not sharded).
    model_mem_bytes = (
        (attn_mem + routed_mem + shared_mem + dense_ffn_mem + embed_mem) / tp
        + ln_mem
        + router_mem
    )
    return model_mem_bytes / 1e9


def calculate_zero3_transient_gb(
    variant,
    experts_per_gpu,
    param_bytes,
    vocab,
    layer_norm,
    ffn_weight_matrices,
    max_live_params,
    prefetch_bucket_params,
    tp=1,
):
    """
    Calculate the ZeRO-3 transient all-gather buffer in GB.

    ZeRO-3 shards resident parameters to Psi/Nd, but a module cannot EXECUTE while
    sharded. DeepSpeed all-gathers each module's FULL parameters into a transient
    buffer immediately before it runs and frees them afterwards, so real peak
    memory exceeds the ZeRO paper's resident-memory model by the size of the
    largest in-flight gather plus whatever is prefetched alongside it.

    The floor is the largest single gathered unit, because that module cannot run
    without being gathered whole. Prefetching adds on top of it, bounded by
    DeepSpeed's live-parameter budget:

        live = max(largest_unit, min(max_live_params, largest_unit + prefetch))

    The gathered units are sized from the same architecture fields, and with the
    same TP treatment, as calculate_model_memory_gb: attention, expert, shared
    expert, dense FFN and embedding weights are TP-sharded, while layer norm and
    router weights are small and replicated.

    Args:
        variant: Variant dictionary with architecture specs
        experts_per_gpu: Number of experts per GPU (already EP-divided)
        param_bytes: Bytes per parameter (2 for BF16, 1 for FP8)
        vocab: Vocabulary size fallback; a variant's own "vocab" key wins
        layer_norm: Layer norm constant (typically 2)
        ffn_weight_matrices: FFN weight matrices constant (typically 3)
        max_live_params: DeepSpeed stage3_max_live_parameters, in ELEMENTS
        prefetch_bucket_params: DeepSpeed stage3_prefetch_bucket_size, in ELEMENTS
        tp: Tensor parallelism degree (default 1)

    Returns:
        dict: {"transient_gb", "largest_unit_params", "largest_unit",
               "live_params", "prefetch_capped"}
    """
    d = variant["d"]
    is_moe = variant.get("expert_ffn", 0) > 0
    has_dense = variant.get("dense_layers", variant["layers"]) > 0

    units = {}

    # A dense transformer layer: attention + SwiGLU FFN + layer norms.
    if has_dense:
        units["dense_layer"] = (
            (variant["attn_per_layer"] + ffn_weight_matrices * d * variant["dense_ffn"]) / tp
            + layer_norm * d
        )
        # On a hybrid family the linear-attention layer is a DIFFERENT and
        # typically LARGER gathered unit (1.10x-1.24x full attention), so it can
        # be the one that sets the floor.
        if variant.get("linear_attn_per_layer"):
            units["dense_linear_attn_layer"] = (
                (variant["linear_attn_per_layer"]
                 + ffn_weight_matrices * d * variant["dense_ffn"]) / tp
                + layer_norm * d
            )

    # An MoE layer: attention + this rank's experts + shared expert + router.
    # experts_per_gpu is used (not the global expert count) because EP has already
    # placed only that many experts on this rank -- ZeRO-3 gathers the rank's set.
    if is_moe:
        moe_body = (
            experts_per_gpu * variant["expert_each"] + variant.get("shared_each", 0)
        )
        units["moe_layer"] = (
            (variant["attn_per_layer"] + moe_body) / tp
            + variant.get("router_each", 0) + layer_norm * d
        )
        if variant.get("linear_attn_per_layer"):
            units["moe_linear_attn_layer"] = (
                (variant["linear_attn_per_layer"] + moe_body) / tp
                + variant.get("router_each", 0) + layer_norm * d
            )

    # One embedding table. For a small model this EXCEEDS a transformer layer --
    # Llama-3.2-3B's 128256 x 3072 table is larger than any of its 28 layers --
    # so the largest gathered unit is not always a layer.
    units["embedding"] = variant.get("vocab", vocab) * d / tp

    largest_unit = max(units, key=units.get)
    largest_unit_params = units[largest_unit]

    with_prefetch = largest_unit_params + prefetch_bucket_params
    live_params = max(largest_unit_params, min(max_live_params, with_prefetch))

    return {
        "transient_gb": live_params * param_bytes / 1e9,
        "largest_unit_params": largest_unit_params,
        "largest_unit": largest_unit,
        "live_params": live_params,
        "prefetch_capped": with_prefetch > max_live_params,
    }


def compute_attention_flops_dim(variant):
    """Effective hidden width for the O(seq_len^2) attention FLOPs term.

    NOT the same quantity as the model's `d`, and using `d` understates the term
    for 11 of 22 catalog variants.

    The two score/context matmuls are parameter-free, so 6ND does not cover them
    and they are added separately. Per layer per sequence, forward:
        QK^T          2 * s^2 * q_heads * qk_head_dim
        softmax(.)V   2 * s^2 * q_heads * v_head_dim
    and backward is ~2x forward, so fwd+bwd is
        6 * s^2 * q_heads * (qk_head_dim + v_head_dim).

    ATTENTION_FLOPS_FACTOR is 12 against a single width, i.e. 12 * s^2 * W. So
    W = q_heads * (qk_head_dim + v_head_dim) / 2, which this returns.

    For standard MHA/GQA where qk_head_dim == v_head_dim == d / q_heads this
    reduces to exactly `d`, so the 11 conforming variants are unaffected. It
    differs in two cases:

    1. A STORED head_dim that is not d / q_heads. Seven variants: Qwen3-32B
       (1.60x), Qwen3-30B-A3B (2.00x), Qwen3-235B-A22B (2.00x), Qwen3.5-27B
       (1.20x), Qwen3.5-35B-A3B (2.00x), Qwen3.5-122B-A10B (2.67x) and
       Qwen3.5-397B-A17B (2.00x). Note GQA does not cause this -- the score and
       context matmuls run over q_heads regardless of kv_heads, because each query
       head attends with its own (s, s) matrix.
    2. MLA, where qk and v head dims differ FROM EACH OTHER as well as from
       d / q_heads. DeepSeek-V3 and V3.1 publish qk_nope_head_dim 128 +
       qk_rope_head_dim 64 = 192 and v_head_dim 128 over 128 heads against
       d = 7168, giving W = 128 * (192 + 128) / 2 = 20480, a 2.86x understatement.

    Falls back to d / q_heads when a variant stores no head dims, which reproduces
    the old `d` behaviour. That fallback is a KNOWN GAP, not a verified value: it
    currently applies to DeepSeek-V4, whose config is unpublished, so its attention
    FLOPs are very likely understated by about the same factor as its siblings.
    Do not close that by copying V3's numbers across.
    """
    d = variant["d"]
    q_heads = variant.get("q_heads") or 0
    if q_heads <= 0:
        return d

    qk = variant.get("qk_head_dim")
    v = variant.get("v_head_dim")
    if qk is None or v is None:
        head_dim = variant.get("head_dim") or (d // q_heads)
        qk = qk if qk is not None else head_dim
        v = v if v is not None else head_dim

    return q_heads * (qk + v) / 2.0


def calculate_training_time_months(
    total_tokens, active_params_b, num_gpus, peak_tflops_per_gpu, mfu, efficiency,
    layers=0, hidden_dim=0, seq_len=0, attention_flops_factor=0,
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
        layers: Model layers (for the attention term; 0 disables it)
        hidden_dim: Model hidden dimension d (for the attention term; 0 disables it)
        seq_len: Sequence length (for the attention term; 0 disables it)
        attention_flops_factor: FLOPs factor for the O(seq_len^2) attention
            matmuls across fwd+bwd (e.g. 12). 0 disables the attention term.

    Returns:
        float: Training time in months
    """
    # Linear-in-tokens compute: 6 * tokens * parameters
    # Factor of 6: 1 forward + 2 backward (gradients).
    linear_flops = 6 * total_tokens * active_params_b * 1e9

    # Attention compute: O(seq_len^2) score (QK^T) + context (softmax*V) matmuls,
    # not captured by 6ND. Per sequence: factor * layers * seq_len^2 * d.
    # Over all sequences (num_sequences = total_tokens / seq_len) this becomes
    # factor * layers * seq_len * total_tokens * d — linear in seq_len, so a
    # 32K-context run costs more attention FLOPs than a 4K run at equal tokens.
    attention_flops = 0.0
    if attention_flops_factor and layers and hidden_dim and seq_len:
        attention_flops = (
            attention_flops_factor * layers * seq_len * total_tokens * hidden_dim
        )

    total_flops = linear_flops + attention_flops

    # Achieved TFLOPs/s across all GPUs
    achieved_tflops_per_sec = peak_tflops_per_gpu * num_gpus * mfu * efficiency

    # Training time in seconds
    training_time_seconds = total_flops / (achieved_tflops_per_sec * 1e12)

    # Convert to months (30.44 days per month average)
    training_time_days = training_time_seconds / (24 * 3600)
    training_time_months = training_time_days / 30.44

    return training_time_months


def calculate_training_time_months_with_overhead(
    total_tokens,
    active_params_b,
    num_gpus,
    peak_tflops_per_gpu,
    mfu,
    efficiency,
    tok_batch,
    *,
    bubble_fraction=0.0,
    exposed_comm_ms_per_step=0.0,
    layers=0,
    hidden_dim=0,
    seq_len=0,
    attention_flops_factor=0,
):
    """Training time in months including PP bubble and exposed comm overhead.

    Model:
      - compute time = ideal FLOPs-based time (same as
        calculate_training_time_months).
      - bubble = idle pipeline time; inflates the compute time by
        bubble_fraction/(1 - bubble_fraction) seconds.
      - comm   = absolute exposed communication, added as
        (exposed_comm_ms_per_step / 1000) * steps seconds.

    Reduces exactly to the FLOPs-only estimate when bubble_fraction == 0 and
    exposed_comm_ms_per_step == 0.

    Args:
        total_tokens: Total training tokens.
        active_params_b: Active parameters in billions.
        num_gpus: Number of GPUs.
        peak_tflops_per_gpu: Peak TFLOPS per GPU for the precision.
        mfu: Model FLOPs Utilization (fraction of peak).
        efficiency: ZeRO/efficiency multiplier.
        tok_batch: Tokens per global batch (used to derive steps).
        bubble_fraction: Pipeline bubble as a fraction (0..1) of compute time.
        exposed_comm_ms_per_step: Exposed (non-overlapped) comm ms per step.

    Returns:
        dict: est_months, compute_months, bubble_months, comm_months,
              bubble_pct (of total), comm_pct (of total).
    """
    _SECONDS_PER_MONTH = 24 * 3600 * 30.44

    # Ideal compute time (FLOPs-based), in seconds.
    compute_months = calculate_training_time_months(
        total_tokens, active_params_b, num_gpus, peak_tflops_per_gpu, mfu, efficiency,
        layers=layers, hidden_dim=hidden_dim, seq_len=seq_len,
        attention_flops_factor=attention_flops_factor,
    )
    compute_seconds = compute_months * _SECONDS_PER_MONTH

    # Bubble inflates compute time. Clamp fraction to avoid blow-up at ~1.0.
    bf = max(0.0, min(0.99, bubble_fraction))
    bubble_seconds = compute_seconds * bf / (1.0 - bf) if bf > 0 else 0.0

    # Exposed comm added as absolute ms/step * number of steps.
    steps = total_tokens / tok_batch if tok_batch > 0 else 0
    comm_seconds = (exposed_comm_ms_per_step / 1000.0) * steps

    total_seconds = compute_seconds + bubble_seconds + comm_seconds

    est_months = total_seconds / _SECONDS_PER_MONTH
    bubble_months = bubble_seconds / _SECONDS_PER_MONTH
    comm_months = comm_seconds / _SECONDS_PER_MONTH

    bubble_pct = (bubble_seconds / total_seconds * 100) if total_seconds > 0 else 0.0
    comm_pct = (comm_seconds / total_seconds * 100) if total_seconds > 0 else 0.0

    return {
        "est_months": est_months,
        "compute_months": compute_months,
        "bubble_months": bubble_months,
        "comm_months": comm_months,
        "bubble_pct": bubble_pct,
        "comm_pct": comm_pct,
    }


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

    # Calculate parameter, gradient and optimizer memory based on ZeRO strategy.
    #
    # param_mem is the RESIDENT parameter memory. ZeRO-1 and ZeRO-2 keep a full
    # replica on every rank; only ZeRO-3 (Pos+g+p) partitions it. ZeRO paper
    # (arXiv:1910.02054v3) sec 5.3: "With parameter partitioning, we reduce the
    # memory consumption of an Psi parameter model from 16*Psi to 16*Psi/Nd."
    param_mem = model_mem_gb
    zero3_transient = 0.0
    zero3_transient_detail = None
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
        # Parameters are sharded too, using the same dense/MoE split as the
        # optimizer above: MoE experts are only replicated within dp/ep ranks.
        #
        # Was absent entirely, so ZeRO-3 total memory came out BYTE-IDENTICAL to
        # ZeRO-2 -- it paid ZeRO-3's 0.8 efficiency multiplier while delivering
        # none of its parameter saving, and so could never be the right choice.
        #
        # LIMITATION -- param_mem above is the paper's RESIDENT-memory model. A
        # sharded layer cannot execute, so DeepSpeed all-gathers each module whole
        # into a transient buffer just before it runs; that buffer is added below
        # as zero3_transient. What remains unmodelled is activation-side effects of
        # the gather schedule, so the figure is still not a guaranteed peak.
        param_dense = model_mem_gb * dense_frac / dp
        param_moe = model_mem_gb * moe_frac / eff_dp_moe
        param_mem = param_dense + param_moe

        # Transient all-gather buffer. Constants and TP come from the wildcard
        # config import, the same way EP is read in calculate_model_memory_gb.
        zero3_transient_detail = calculate_zero3_transient_gb(
            variant=variant,
            experts_per_gpu=experts_per_gpu,
            param_bytes=param_bytes,
            vocab=VOCAB,
            layer_norm=LAYER_NORM,
            ffn_weight_matrices=FFN_WEIGHT_MATRICES,
            max_live_params=ZERO3_MAX_LIVE_PARAMS,
            prefetch_bucket_params=ZERO3_PREFETCH_BUCKET_PARAMS,
            tp=TP,
        )
        zero3_transient = zero3_transient_detail["transient_gb"]
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
    # param_mem, not model_mem_gb: they differ only under ZeRO-3, which shards the
    # resident parameters. "model" reports what is actually resident on the GPU.
    #
    # zero3_transient is 0.0 for ZeRO-1 and ZeRO-2, which never shard parameters
    # and so never gather them back.
    raw_total = (
        param_mem + grad_mem + optim_mem + activation_mem + buffer_mem + zero3_transient
    )
    total_mem = raw_total * fragmentation_factor

    return {
        "total": total_mem,
        "model": param_mem,
        "grad": grad_mem,
        "optim": optim_mem,
        "activation": activation_mem,
        "buffer": buffer_mem,
        "zero3_transient": zero3_transient,
        "zero3_transient_detail": zero3_transient_detail,
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
            seq_len=variant.get("seq_len", SEQ_LEN),
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
