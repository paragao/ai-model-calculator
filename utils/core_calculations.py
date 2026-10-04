"""
Core calculation functions shared across multiple analysis phases.

Contains fundamental model memory, training time, and memory breakdown calculations
used by multiple phases of the analysis pipeline.
"""

# Import configurations
from configuration.project_config import *
from configuration.advanced_config import *


def resolve_experts_per_gpu(variant, experts_per_gpu, ep=None):
    """Experts resident on one GPU for this variant, overriding the global count.

    experts_per_gpu arrives from the caller as the GLOBAL N_EXPERTS // EP, which is
    right only for a model that actually has N_EXPERTS experts. A variant declaring
    its own count overrides it, because the global is wrong in both directions:
    Llama-4-Scout has 16 experts (the global 128 overstates it 8x), while
    Qwen3.5-397B has 512 and the DeepSeek entries 256 (understated 4x and 2x).

    Extracted so calculate_model_memory_gb and calculate_memory_for_micro cannot
    disagree. They previously did: the memory function applied this override while
    calculate_memory_for_micro used the raw global for its moe_frac split and its
    MoE dispatch buffer, despite carrying a comment saying it "must match".
    Measured, that affected 7 of 11 MoE variants.
    """
    variant_experts = variant.get("num_experts")
    effective_ep = EP if ep is None else ep
    if variant_experts and effective_ep > 0:
        return max(1, variant_experts // effective_ep)
    return experts_per_gpu


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
        param_bytes: Bytes per parameter AS RESIDENT IN MEMORY. Callers pass
            PARAM_STORAGE_BYTES (2 at either precision), NOT PARAM_BYTES:
            FP8 is a compute dtype and the weights stay BF16.
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
    # for models that actually have N_EXPERTS experts. Resolved by the shared
    # helper so calculate_memory_for_micro applies the identical override.
    # Note moe_layer_params in the catalog already reflects the true count, but
    # it is metadata -- the memory path below uses expert_each * experts_per_gpu.
    experts_per_gpu = resolve_experts_per_gpu(variant, experts_per_gpu, ep)

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
        param_bytes: Bytes per parameter AS RESIDENT IN MEMORY. Callers pass
            PARAM_STORAGE_BYTES (2 at either precision), NOT PARAM_BYTES:
            FP8 is a compute dtype and the weights stay BF16.
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
    for 10 of 22 catalog variants.

    The two score/context matmuls are parameter-free, so 6ND does not cover them
    and they are added separately. Per layer per sequence, forward:
        QK^T          2 * s^2 * q_heads * qk_head_dim
        softmax(.)V   2 * s^2 * q_heads * v_head_dim
    and backward is ~2x forward, so fwd+bwd is
        6 * s^2 * q_heads * (qk_head_dim + v_head_dim).

    ATTENTION_FLOPS_FACTOR is 12 against a single width, i.e. 12 * s^2 * W. So
    W = q_heads * (qk_head_dim + v_head_dim) / 2, which this returns.

    For standard MHA/GQA where qk_head_dim == v_head_dim == d / q_heads this
    reduces to exactly `d`, so the 12 conforming variants are unaffected. It
    differs in two cases:

    1. A STORED head_dim that is not d / q_heads. Seven variants: Qwen3-32B
       (1.60x), Qwen3-30B-A3B (2.00x), Qwen3-235B-A22B (2.00x), Qwen3.5-27B
       (1.20x), Qwen3.5-35B-A3B (2.00x), Qwen3.5-122B-A10B (2.67x) and
       Qwen3.5-397B-A17B (2.00x). Note GQA does not cause this -- the score and
       context matmuls run over q_heads regardless of kv_heads, because each query
       head attends with its own (s, s) matrix.
    2. MLA, where qk and v head dims differ FROM EACH OTHER as well as from
       d / q_heads. DeepSeek-V3, V3.1 and V3.2 each publish qk_nope_head_dim 128
       + qk_rope_head_dim 64 = 192 and v_head_dim 128 over 128 heads against
       d = 7168, giving W = 128 * (192 + 128) / 2 = 20480, a 2.86x understatement.
       All three were checked individually rather than assumed to match.

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


def resolve_recompute_scale(
    layers_per_gpu,
    vp,
    recompute_granularity,
    recompute_num_layers,
    empirical_act_multiplier,
    recompute_boundary_bytes_per_element,
    act_precision_saving_fraction=0.0,
):
    """Fraction of layers recomputed, and the resulting per-layer activation scale.

    Megatron's recompute_method="block" with recompute_granularity="full" recomputes
    recompute_num_layers layers of EVERY pipeline chunk. Under an interleaved
    schedule a chunk is layers_per_gpu / vp layers, so the recomputed FRACTION
    depends on PP and VP, not on recompute_num_layers alone.

    A stored layer costs empirical_act_multiplier bytes per element. A recomputed
    layer stores only its block input, i.e. recompute_boundary_bytes_per_element.

    act_precision_saving_fraction is the share of a STORED layer's bytes that FP8
    saves. It applies to the stored portion ONLY, which is what makes the two
    techniques substitutes rather than additive: a recomputed layer stores no
    activations, so there is nothing for FP8 to halve, and the saving in bytes comes
    out as empirical_act_multiplier * (1 - fraction) * act_precision_saving_fraction.
    At the shipped constants that is 3.5 * (1 - f) B/element, which is what jobs 350
    and 352/354/361/363 x 365/353/362/364 measured. See
    ACT_PRECISION_SAVING_FRACTION in advanced_config.

    Returns (fraction_recomputed, scale) where scale multiplies the per-layer
    activation cost. With no recompute and no precision saving the fraction is 0.0
    and the scale is exactly 1.0, so every caller is unchanged by default.

    This is shared so the base activation term and the VP in-flight term cannot
    drift apart -- the same failure mode that made moe_frac disagree with itself.
    """
    stored_scale = 1.0 - act_precision_saving_fraction
    if not recompute_granularity or recompute_num_layers <= 0:
        # Nothing recomputed, so every layer is stored and every layer saves.
        # NOT 1.0: returning 1.0 here would make the precision saving reachable
        # only when a recipe also enables recompute, which is backwards.
        return 0.0, stored_scale
    if recompute_granularity != "full":
        raise ValueError(
            f"unsupported recompute_granularity {recompute_granularity!r}; "
            "only None and 'full' are modelled"
        )
    if empirical_act_multiplier <= 0:
        raise ValueError("empirical_act_multiplier must be positive")

    layers_per_chunk = layers_per_gpu / vp if vp and vp > 1 else layers_per_gpu
    if layers_per_chunk <= 0:
        return 0.0, stored_scale

    recomputed = min(recompute_num_layers, layers_per_chunk)
    fraction = recomputed / layers_per_chunk
    # Blend the stored and recomputed per-layer costs, expressed as a multiplier on
    # the stored cost so it composes with the existing expressions unchanged. The
    # precision saving multiplies the stored term and NOT the boundary term: a
    # recomputed block's retained input is the next block's input, held in BF16.
    scale = (
        (1.0 - fraction) * stored_scale
        + fraction * recompute_boundary_bytes_per_element / empirical_act_multiplier
    )
    return fraction, scale


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
    cp=1,
    tp=1,
    act_bytes_per_element=2,
    recompute_granularity=None,
    recompute_num_layers=0,
    recompute_boundary_bytes_per_element=2,
    act_precision_saving_fraction=0.0,
    nccl_ep_scaling_factor=0.0,
    fragmentation_factor=1.0,
    experts_per_gpu=0,
    grad_bytes=1.0,
    vocab=None,
    logits_bytes_per_element=0,
    logits_fused_ce=True,
    logits_chunk_tokens=1024,
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
        seq_len: Sequence length. The FULL sequence; cp divides it below, so
            callers pass the model's sequence length, not a per-rank slice.
        cp: Context parallelism degree. Shards the sequence across cp ranks, so
            every token-count-dependent term sees seq_len/cp per GPU.
        act_bytes_per_element: Bytes per activation element for the MoE EXPERT
            intermediate only. This is the BF16 per-element size of that tensor; the
            FP8 saving is applied separately via act_precision_saving_fraction, which
            now scales the expert half as well as the dense half. It used to be
            described as deliberately precision-invariant, on the grounds that the
            FP8 saving was measured on a dense model; jobs 376/377 measured it on a
            100% MoE model and refuted that. See ACT_BYTES_PER_ELEMENT in
            advanced_config.
        act_precision_saving_fraction: Share of a STORED layer's activation bytes
            that FP8 saves, applied to BOTH the d-proportional dense half and the
            MoE expert half. 0.0 at BF16, so the default is exactly
            inert. Applies to the stored portion only, which makes FP8 and
            recompute SUBSTITUTES -- see resolve_recompute_scale and
            ACT_PRECISION_SAVING_FRACTION, measured at 5.5 of 32 B/element
            (17.19%). It read 3.5 of 12 (29.17%) until the multiplier was
            re-measured; both the base and the saved bytes come from the same real
            pretraining pair now.
        recompute_granularity: None (store every layer's activations, the default
            and the behaviour before this existed) or "full" (recompute whole
            blocks, as recompute_granularity="full" in a Megatron/NeMo recipe).
        recompute_num_layers: Layers recomputed per PIPELINE CHUNK, matching
            Megatron's recompute_num_layers under recompute_method="block". The
            recomputed FRACTION is this over layers_per_gpu/vp, so it depends on
            PP and VP as well -- see resolve_recompute_scale.
        recompute_boundary_bytes_per_element: Bytes a recomputed block still stores
            for its own input. Applies to the base term AND the VP in-flight term,
            which previously took no recompute reduction at all.
        topk: Top-K for MoE routing
        param_bytes: Bytes per parameter AS RESIDENT IN MEMORY -- callers pass
            PARAM_STORAGE_BYTES, not PARAM_BYTES. Every remaining use here wants
            the resident size: the moe_frac split (which must match
            calculate_model_memory_gb), the ZeRO-3 all-gather buffer (which
            gathers resident weights), and the a2a/dispatch/PP buffers (which
            carry activations, held at BF16 for the same reason as
            act_bytes_per_element).
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
        grad_bytes: MULTIPLIER on parameter memory for gradients, the same shape as
            optim_bytes. 1.0 keeps gradients at PARAM_BYTES, which is wrong for FP8
            because gradients reduce in BF16 or higher; pass
            GRAD_BYTES_PER_PARAM / PARAM_BYTES. Defaults to 1.0 so an existing
            caller that does not pass it keeps its previous behaviour.
        vocab: Vocabulary size for the output-logits term. A variant's own "vocab"
            key wins over this. None disables the term.
        logits_bytes_per_element: Bytes per logit element, covering the bf16 logits
            plus the fp32 softmax and its gradient. 0 disables the term, which is
            the default so an existing caller is unaffected.
        logits_fused_ce: When True, only logits_chunk_tokens rows are live at once.
        logits_chunk_tokens: Rows materialised at once under a fused CE.

    Returns:
        dict: Memory breakdown with keys:
            - total: Total memory in GB
            - model: Model parameter memory
            - grad: Gradient memory
            - optim: Optimizer state memory
            - activation: Activation memory
            - buffer: Communication buffer memory
            - logits: Output-logits memory (0 when the term is disabled)
    """
    d = variant["d"]
    is_moe = variant.get("expert_ffn", 0) > 0

    # Context parallelism shards the SEQUENCE across cp ranks, so each GPU holds
    # seq_len/cp tokens. Applied once here rather than at each site, because every
    # seq_len consumer below -- layer activations, the VP in-flight overhead, the
    # a2a and MoE dispatch buffers, the PP send/recv buffers and the output logits
    # -- is a per-GPU token-count term and all of them shard alike.
    #
    # Before this, cp reached the model ONLY through the caller's
    # dp = gpus // (TP * PP * CP), so raising cp SHRANK dp and therefore RAISED
    # reported per-GPU memory, while its actual purpose is to cut activation
    # memory. Memory moved the wrong way as cp rose, the same class of defect as
    # the moe_frac TP omission above.
    seq_len = seq_len / max(1, cp)

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

    # Resolve the expert count the SAME way calculate_model_memory_gb does, so
    # moe_frac below is a fraction of the model memory that function returned.
    # Using the raw global here was wrong for 7 of 11 MoE variants.
    experts_per_gpu = resolve_experts_per_gpu(variant, experts_per_gpu, ep)

    # Estimate fraction of model memory that is MoE expert params vs non-MoE
    if is_moe and model_mem_gb > 0:
        # Expert memory fraction based on architecture. This MUST match
        # calculate_model_memory_gb, which divides routed weights by TP -- without
        # that division the ratio exceeds 1.0 for any TP > 1 and clamps, making
        # dense_frac 0 and pushing ALL optimizer and ZeRO-3 parameter sharding onto
        # eff_dp_moe. Measured on Qwen3-235B-A22B at TP=2, that raised reported
        # optimizer memory from 5.36 to 8.04 GB even though TP had halved the
        # weights -- memory moving the wrong way as TP rises.
        routed_mem = (
            experts_per_gpu * moe_layers_per_gpu * variant["expert_each"]
            * param_bytes / max(1, TP)
        )
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
        # grad_bytes is a multiplier on parameter memory, so gradients can be held
        # at a precision independent of PARAM_BYTES. 1.0 means "same as params".
        grad_mem = model_mem_gb * grad_bytes
        # Optimizer: dense params sharded across full DP, MoE params across eff_dp_moe
        optim_dense = model_mem_gb * dense_frac * optim_bytes / dp
        optim_moe = model_mem_gb * moe_frac * optim_bytes / eff_dp_moe
        optim_mem = optim_dense + optim_moe
    elif zero_strategy == 2:
        grad_mem = model_mem_gb * grad_bytes / dp
        optim_dense = model_mem_gb * dense_frac * optim_bytes / dp
        optim_moe = model_mem_gb * moe_frac * optim_bytes / eff_dp_moe
        optim_mem = optim_dense + optim_moe
    elif zero_strategy == 3:
        grad_mem = model_mem_gb * grad_bytes / dp
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

    # --- Output-logits memory ---
    #
    # The LM head produces a (micro, seq_len, vocab) tensor and the cross-entropy
    # backward needs its gradient. This is NOT covered by the activation term
    # below, which is per-layer in d and never touches vocab.
    #
    # Under a fused or chunked CE only logits_chunk_tokens rows are live at once, so
    # above the chunk size this term stops scaling with micro. Unfused it grows
    # linearly with micro and becomes the single largest term in this function.
    #
    # The LM head WEIGHT gradient is not double-counted here: the head's parameters
    # are already in model_mem_gb via embed_mem, so grad_mem covers them.
    logits_mem = 0.0
    logits_vocab = variant.get("vocab") or vocab
    if logits_vocab and logits_bytes_per_element:
        logits_tokens = micro * seq_len
        live_tokens = (min(logits_tokens, logits_chunk_tokens)
                       if logits_fused_ce else logits_tokens)
        logits_mem = live_tokens * logits_vocab * logits_bytes_per_element / 1e9

    # --- Activation memory ---
    # Split into two parts whose FORMULAS differ, rather than one aggregate. This
    # sentence read "the two parts that behave DIFFERENTLY under precision" until
    # jobs 376/377, which is now contradicted three lines below -- they behave the
    # SAME under precision and differ only in how each is computed:
    #
    #   dense part  -- the d-proportional per-layer cost (attention and MLP
    #                  intermediates). Present on EVERY layer, dense or MoE alike.
    #                  Measured to shrink under FP8; takes act_precision_saving.
    #   expert part -- the MoE routed-expert intermediate, on MoE layers only.
    #                  ALSO measured to shrink under FP8 (jobs 376/377,
    #                  Qwen3-30B-A3B: 29.505% measured). Takes the SAME saving.
    #
    #                  The "indistinguishable from the dense half" reading of that
    #                  pair was CONDITIONAL ON THE OLD 12 B/element multiplier,
    #                  which made symmetry predict 29.167%. At the measured 32 the
    #                  same symmetry predicts 17.19% for that run, so it now
    #                  under-predicts by 12.3 pp. Symmetry is still the better of
    #                  the two options and stands -- invariance predicts 12.50%,
    #                  further away -- but see ACT_PRECISION_SAVED_BYTES_PER_ELEMENT
    #                  in advanced_config for what the residual now implies.
    #
    # The two are still computed separately because their FORMULAS differ -- the
    # dense half is d * multiplier and the expert half is topk * expert_ffn * 2 *
    # act_bytes_per_element -- and because the recompute blend may yet be shown to
    # differ between them. They no longer differ in PRECISION.
    #
    # Algebraically identical to the single aggregate this replaces: that branch
    # computed moe_layers*(d*m + e) + (L - moe_layers)*d*m, which is L*d*m +
    # moe_layers*e. The whole 704-pair sweep is byte-identical at the inert default,
    # and a test pins the equivalence.
    act_dense_full = layers_per_gpu * seq_len * micro * d * empirical_act_multiplier
    act_expert_full = 0.0
    if is_moe and moe_layers_per_gpu > 0:
        # Expert intermediate = tokens * topk * expert_ffn * 2 (input + output)
        #
        # KNOWN GAP, pre-existing and NOT addressed here: `topk` arrives from the
        # global project_config.TOPK (= 8), not from the variant's own `topk` field,
        # so 3 of 11 MoE variants are computed at the wrong routing width --
        # Llama-4-Scout-17B-16E and Llama-4-Maverick-17B-128E carry topk=1 and
        # Qwen3.5-397B-A17B carries topk=10 (DeepSeek-V4 carries none at all). The
        # same global reaches the routing-buffer and all-to-all terms below. Found
        # while validating the symmetry fix: recovering the implied expert share
        # algebraically from a before/after sweep agreed with the catalog EXACTLY on
        # every variant whose own topk is 8 and disagreed on precisely those three.
        # Fixing it changes memory for those variants, so it is a separate change
        # with its own baselines rather than a drive-by edit here.
        expert_ffn = variant.get("expert_ffn", 0)
        act_expert_full = (
            moe_layers_per_gpu * seq_len * micro
            * topk * expert_ffn * 2 * act_bytes_per_element
        )

    # Activation recomputation and FP8 precision saving. A recomputed block stores
    # only its input and an FP8-stored layer costs less, so both scale the per-layer
    # cost and they COMPOSE rather than add -- see resolve_recompute_scale. Inert
    # (both scales exactly 1.0) unless a recipe's recompute settings are supplied or
    # PRECISION is FP8. Derived once and applied to BOTH the base term and the VP
    # in-flight term below, so the two cannot disagree.
    recompute_fraction, dense_act_scale = resolve_recompute_scale(
        layers_per_gpu,
        vp,
        recompute_granularity,
        recompute_num_layers,
        empirical_act_multiplier,
        recompute_boundary_bytes_per_element,
        act_precision_saving_fraction,
    )
    # Same recompute blend AND the same precision saving. Taken from the SAME helper
    # rather than written out here, so neither the recompute halves nor the precision
    # halves can drift apart.
    #
    # This call passed 0.0 as the saving until the MoE pair below. The expert
    # intermediate was held precision-invariant because the 3.5 B/element figure was
    # measured on a DENSE model (Llama-3.1-8B), and extending it to experts by
    # symmetry would have invented a constant for an unmeasured term.
    #
    # MEASURED, jobs 376/377 -- Qwen3-30B-A3B (100% MoE, 48 layers), 16 GPUs, EP=8,
    # recompute off and CUDA graphs forced off on BOTH arms, resolved configs
    # differing only in precision fields:
    #
    #     ACT = peak max-allocated - steady mem-allocated
    #     bf16  68.652 - 52.363 = 16.289 GB
    #     fp8   63.847 - 52.364 = 11.483 GB   -> saving 29.505%
    #
    # The model was chosen because its two activation halves were EXACTLY equal AT
    # THE MULTIPLIER THEN SHIPPED -- dense d*m = 2048*12 = 24576 B and expert
    # topk*e_ffn*2*bytes = 8*768*2*2 = 24576 B -- so the two hypotheses differed by
    # exactly a factor of two:
    #
    #     experts precision-INVARIANT (what this line used to encode):  14.583%
    #     experts take the SAME saving as dense:                        29.167%
    #
    # Measured 29.505% sat 0.338 pp from the symmetric prediction and 14.921 pp
    # from the invariant one, against a refutation threshold of 21.875% recorded
    # before the jobs were submitted. So the asymmetry is refuted and the expert
    # half takes the saving.
    #
    # THAT NEAR-EXACT AGREEMENT DID NOT SURVIVE THE MULTIPLIER BEING MEASURED.
    # At the corrected 32 B/element the halves are no longer equal -- dense
    # 2048*32 = 65536 B against the same expert 24576 B, a 72.7/27.3 split -- and
    # symmetry now predicts 17.19% for that run against the same measured 29.505%,
    # i.e. UNDER by 12.3 pp where it once looked 0.34 pp over. The decision stands
    # because invariance is worse still (12.50%), and because the correction moved
    # the TOTAL much closer: 44.0 B/element predicted against 40.45 measured, where
    # the old constants predicted 24.0, low by 40.7%.
    #
    # The expert half gets the SAME constant rather than its own. Holding the dense
    # half at its measured 17.19%, this pair implies an expert-side saving of 62.4%
    # -- which EXCEEDS the 50% ceiling an FP8 storage swap can deliver, so the
    # residual is more likely an under-counted expert ELEMENT count than a missing
    # expert saving constant. `peak - steady` also contains the FP8 parameter
    # all-gather transient, present only in the fp8 arm, so 29.505% is itself a
    # LOWER bound. Sign and order are settled; the magnitude is not, and no second
    # constant is invented here. Pinning it needs a pair with fp8_param_gather
    # disabled on the fp8 arm, plus a check of the expert element count against a
    # gate/up/down decomposition.
    #
    # Note the RECOMPUTE blend is still shared between the halves and is NOT
    # validated by this pair, which ran at f=0. A recomputed MoE layer arguably
    # stores no expert intermediate at all, while this helper charges it the same
    # boundary-tensor blend as the dense half. That is a separate, unmeasured
    # question -- deliberately left alone here rather than changed by inference.
    _, expert_act_scale = resolve_recompute_scale(
        layers_per_gpu,
        vp,
        recompute_granularity,
        recompute_num_layers,
        empirical_act_multiplier,
        recompute_boundary_bytes_per_element,
        act_precision_saving_fraction,
    )

    activation_mem = (
        act_dense_full * dense_act_scale + act_expert_full * expert_act_scale
    ) * selective_act_checkpointing_multiplier / 1e9

    # VP activation overhead: interleaved schedule keeps more activations in-flight
    # In interleaved 1F1B, warmup fills (PP-1)*VP micro-batches before steady state.
    # Each micro-batch occupies layers_per_virtual_chunk = layers_per_gpu / VP layers.
    # Total layer-activations stored = (PP-1) * layers_per_gpu (independent of VP).
    # But lower VP = fewer but larger chunks = less overhead from per-chunk metadata.
    # `vp and` guards vp=None, which a real recipe uses to mean "no interleaving"
    # and which raised TypeError here on None > 1 for any direct caller. The
    # internal caller passes VP=1 so the shipped sweep never hit it.
    if vp and vp > 1 and pp > 1:
        layers_per_virtual_chunk = layers_per_gpu // vp
        inflight_micros = (pp - 1) * vp  # warmup micro-batches
        # Same dense/expert split as the base term above, so the precision saving
        # reaches the d-proportional part here too and the expert part stays
        # invariant. In-flight chunks are recomputed on the same schedule as any
        # other, so the recompute blend applies here as well. Before that it did
        # not, which left the VP term as the single largest activation contributor
        # for a recipe that recomputes: at PP=2/VP=20 it was 16.106 of 23.354 GB on
        # Llama-3.1-70B.
        dense_act_per_layer = seq_len * micro * d * empirical_act_multiplier
        expert_act_per_layer = 0.0
        if is_moe and moe_layers_per_gpu > 0:
            expert_ffn = variant.get("expert_ffn", 0)
            expert_act_per_layer = (
                seq_len * micro * topk * expert_ffn * 2 * act_bytes_per_element
            )
        avg_act_per_layer = (
            dense_act_per_layer * dense_act_scale
            + expert_act_per_layer * expert_act_scale
        )
        vp_act_overhead = inflight_micros * layers_per_virtual_chunk * avg_act_per_layer / 1e9
        activation_mem += vp_act_overhead

    # --- Buffer memory ---
    # All-to-all routing buffers.
    #
    # GATED ON is_moe THIS ROUND. A dense model has no router and no expert
    # all-to-all, so it has no routing buffer -- but this term was computed
    # unconditionally from the GLOBAL topk (8), so every dense variant was charged
    # one. It is small but it is not nothing: on a 0.6B model at seq 4096 it is
    # 0.134 GB, and on a dense 70B it is 0.537 GB per micro-batch.
    #
    # Same oversight as the NCCL workspace below and the same evidence for it:
    # moe_dispatch_buf, immediately after, is already gated on `is_moe and
    # experts_per_gpu > 0`. Only this one was not.
    a2a_buf = 0.0
    if is_moe:
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

    # NCCL workspace: base + scaling with EP group size.
    #
    # GATED ON WORLD SIZE. A single-GPU configuration has no communicator, so it
    # pays no NCCL workspace. This term used to be ungated while every other
    # buffer component here was gated on the parallelism that creates it
    # (moe_dispatch_buf on experts_per_gpu, pp_buf on pp > 1), so a 1-GPU run was
    # charged the full nccl_mem_buf -- 6.00 GB, which on a 0.6B model was 34% of
    # the raw total and flipped micro-batch 4 from fitting to OOM.
    #
    # tp is included because tp > 1 is by definition multi-GPU; it reaches this
    # function only because this round started forwarding it from
    # calculate_viable_micro_batch_sizes, which had it in scope and dropped it.
    # ep is deliberately NOT a factor: expert parallelism is carved out of the
    # data-parallel group, so ep > 1 already implies dp > 1 and multiplying by it
    # would overcount the world. It still scales the workspace below.
    world_size = dp * tp * pp * cp
    nccl_workspace = 0.0
    if world_size > 1:
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
        param_mem + grad_mem + optim_mem + activation_mem + buffer_mem
        + zero3_transient + logits_mem
    )
    total_mem = raw_total * fragmentation_factor

    return {
        "total": total_mem,
        "model": param_mem,
        "grad": grad_mem,
        "optim": optim_mem,
        "activation": activation_mem,
        "buffer": buffer_mem,
        "logits": logits_mem,
        "recompute_fraction": recompute_fraction,
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
        PARAM_STORAGE_BYTES,
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
            param_bytes=PARAM_STORAGE_BYTES,
            empirical_act_multiplier=EMPIRICAL_ACT_MULTIPLIER,
            selective_act_checkpointing_multiplier=SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER,
            fwd_bwd_routing_buff_passes=FWD_BWD_ROUTING_BUFF_PASSES,
            nccl_mem_buf=NCCL_MEM_BUF,
            optim_bytes=OPTIM_BYTES,
            ep=EP,
            pp=PP,
            vp=VP,
            tp=tp,
            nccl_ep_scaling_factor=NCCL_EP_SCALING_FACTOR,
            fragmentation_factor=FRAGMENTATION_FACTOR,
            experts_per_gpu=experts_per_gpu,
            grad_bytes=GRAD_BYTES,
            vocab=VOCAB,
            logits_bytes_per_element=LOGITS_BYTES_PER_ELEMENT,
            logits_fused_ce=LOGITS_FUSED_CE,
            logits_chunk_tokens=LOGITS_CHUNK_TOKENS,
            cp=CP,
            act_bytes_per_element=ACT_BYTES_PER_ELEMENT,
            recompute_granularity=RECOMPUTE_GRANULARITY,
            recompute_num_layers=RECOMPUTE_NUM_LAYERS,
            recompute_boundary_bytes_per_element=RECOMPUTE_BOUNDARY_BYTES_PER_ELEMENT,
            act_precision_saving_fraction=ACT_PRECISION_SAVING_FRACTION,
        )

        # Check if this micro batch size fits in memory
        if memory_breakdown["total"] <= gpu_mem_gb * threshold:
            viable_micros.append(micro)

    return sorted(viable_micros)
