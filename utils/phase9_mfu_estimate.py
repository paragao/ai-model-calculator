"""
Phase 9: estimated MFU, parallel plan search, throughput, days and cost.

Replaces the fixed project_config.MFU (0.40) for an end-to-end estimate. For one model
on one instance type and GPU count it

  1. enumerates every TP x PP x EP plan that divides the GPUs (Megatron layout,
     GPUs = TP x PP x DP, EP carved out of DP; mfu_model.candidate_plans),
  2. ranks them by estimated throughput (mfu_model.estimate),
  3. checks the best ones against the phase-1 memory model run with THAT plan's TP, PP
     and EP, and scores each viable micro batch,
  4. keeps the fastest plan that fits, and reports MFU, tokens/s, days (90% goodput),
     cost, and every part of the step time.

Every formula, constant and source is in docs/CALCULATOR-MFU.md. The accuracy against
published runs is utils/validation_runs.py (python3 -m utils.validation_runs).

The memory modules read TP/PP/EP/PRECISION from configuration.project_config when they
are imported, so they are reloaded for each candidate plan and restored afterwards: this
function leaves project_config and the other phases exactly as it found them.
"""

import importlib
import math

from utils import mfu_model as mm

# GPU / instance names a user may write -> instance type.
HARDWARE_ALIASES = {
    "p5.48xlarge": "p5.48xlarge", "p5": "p5.48xlarge", "h100": "p5.48xlarge",
    "p5e.48xlarge": "p5e.48xlarge", "p5e": "p5e.48xlarge",
    "p5en.48xlarge": "p5en.48xlarge", "p5en": "p5en.48xlarge", "h200": "p5en.48xlarge",
    "p6-b200.48xlarge": "p6-b200.48xlarge", "p6-b200": "p6-b200.48xlarge", "b200": "p6-b200.48xlarge",
    "p6-b300.48xlarge": "p6-b300.48xlarge", "p6-b300": "p6-b300.48xlarge", "b300": "p6-b300.48xlarge",
    "p4d.24xlarge": "p4d.24xlarge", "p4d": "p4d.24xlarge", "a100": "p4d.24xlarge", "a100-40gb": "p4d.24xlarge",
    "p4de.24xlarge": "p4de.24xlarge", "p4de": "p4de.24xlarge", "a100-80gb": "p4de.24xlarge",
}
GPU_OF_INSTANCE = {
    "p5.48xlarge": "H100 80GB", "p5e.48xlarge": "H200 141GB", "p5en.48xlarge": "H200 141GB",
    "p6-b200.48xlarge": "B200 180GB", "p6-b300.48xlarge": "B300 268GB",
    "p4d.24xlarge": "A100 40GB", "p4de.24xlarge": "A100 80GB",
}
# Memory and bandwidth per instance, in the units of configuration/hardware_config.py
# (tests/test_mfu_model.py checks the shared entries against that file).
INSTANCE_SPECS = {
    "p5.48xlarge": {"mem_gb": 80, "inter_node_bw_gb": 400, "intra_node_bw_gbps": 900, "peak_tflops_bf16": 989},
    "p5e.48xlarge": {"mem_gb": 141, "inter_node_bw_gb": 400, "intra_node_bw_gbps": 900, "peak_tflops_bf16": 989},
    "p5en.48xlarge": {"mem_gb": 141, "inter_node_bw_gb": 400, "intra_node_bw_gbps": 900, "peak_tflops_bf16": 989},
    "p6-b200.48xlarge": {"mem_gb": 180, "inter_node_bw_gb": 400, "intra_node_bw_gbps": 1800, "peak_tflops_bf16": 2250},
    "p6-b300.48xlarge": {"mem_gb": 268, "inter_node_bw_gb": 800, "intra_node_bw_gbps": 1800, "peak_tflops_bf16": 2250},
    "p4d.24xlarge": {"mem_gb": 40, "inter_node_bw_gb": 100, "intra_node_bw_gbps": 600, "peak_tflops_bf16": 312},
    "p4de.24xlarge": {"mem_gb": 80, "inter_node_bw_gb": 100, "intra_node_bw_gbps": 600, "peak_tflops_bf16": 312},
}
# On-demand USD per instance-hour (an estimate, not a quote).
INSTANCE_COST_PER_HOUR = {
    "p5.48xlarge": 98.32, "p5e.48xlarge": 98.32, "p5en.48xlarge": 140.00,
    "p6-b200.48xlarge": 180.00, "p6-b300.48xlarge": 250.00,
    "p4d.24xlarge": 32.77, "p4de.24xlarge": 40.97,
}
MAX_AUTO_GPUS = 4096


def normalise_hardware(value):
    """Instance type for a GPU or instance name ("B300", "p6-b300", "nvidia h100"). Unknown -> ValueError."""
    key = str(value or "").strip().lower().replace("_", "-")
    for prefix in ("nvidia ", "nvidia-", "aws "):
        if key.startswith(prefix):
            key = key[len(prefix):]
    key = key.replace(" ", "-")
    if key in HARDWARE_ALIASES:
        return HARDWARE_ALIASES[key]
    for suffix in ("-sxm", "-ultra", "-gpu", "s"):
        if key.endswith(suffix) and key[: -len(suffix)] in HARDWARE_ALIASES:
            return HARDWARE_ALIASES[key[: -len(suffix)]]
    raise ValueError(
        f"Unknown hardware '{value}'. Supported: "
        + ", ".join(f"{k} ({g})" for k, g in GPU_OF_INSTANCE.items())
    )


def instance_specs(instance):
    return INSTANCE_SPECS.get(instance, INSTANCE_SPECS["p5.48xlarge"])


def instance_cost(instance):
    return INSTANCE_COST_PER_HOUR.get(instance, 98.32)


def shape_from_preset(v, vocab):
    """mfu_model shape for a catalog variant (fields of variants_config)."""
    layers, d = v["layers"], v["d"]
    heads = v.get("q_heads") or max(1, d // 128)
    head_dim = v.get("head_dim") or d // heads
    attn = v["attn_per_layer"]
    dense_layers = v.get("dense_layers", layers)
    moe_layers = layers - dense_layers
    expert_each = v.get("expert_each", 0)
    dense_mlp = v.get("dense_layer_params", attn) - attn
    total = v["total_params_B"] * 1e9
    active = v.get("active_params_B", v["total_params_B"]) * 1e9
    embed = (v.get("vocab") or vocab or 0) * d
    nonexpert = (layers * attn + dense_layers * dense_mlp
                 + moe_layers * (v.get("shared_each", 0) + v.get("router_each", 0)) + 2 * embed)
    expert_total = max(0.0, total - nonexpert) if moe_layers else 0.0
    expert_active = max(0.0, active - nonexpert) if moe_layers else 0.0
    topk = round(expert_active / (moe_layers * expert_each)) if moe_layers and expert_each else 0
    return {
        "layers": layers, "d": d, "moe_layers": moe_layers, "topk": topk, "n_group": 1, "topk_group": topk,
        "expert_ffn": v.get("expert_ffn", 0), "n_experts": v.get("num_experts", 0),
        "attention": {"heads": heads, "kv_heads": v.get("kv_heads") or heads, "d_qk": head_dim, "d_v": head_dim,
                      "full_layers": layers, "linear_layers": 0, "lin_heads": heads, "lin_dim": head_dim},
        "matmul_active": active - embed, "expert_active_matmul": expert_active,
        "total_b": v["total_params_B"], "expert_total_b": expert_total / 1e9,
    }


def estimate_training(config, variant):
    """Plan search + MFU + days + cost for one GPU count.

    config keys: hardware (instance type), num_gpus, precision ("bf16" | "fp8"), sequence_length,
    total_tokens_b, tensor_parallel / pipeline_parallel / expert_parallel (fix a degree; 1 =
    search), virtual_pipeline_parallel, global_batch_size (sequences), micro_batch_size
    (reported only), gpus_per_node, mfu (explicit override; otherwise estimated),
    max_fitting_plans (how many memory-fitting plans to score; default 6).
    Returns the result dict, or {"error": ...}.
    """
    import configuration.project_config as pc
    import configuration.advanced_config as ac
    import utils.core_calculations
    import utils.phase1_memory

    saved = {k: getattr(pc, k) for k in
             ("VOCAB", "TP", "PP", "EP", "CP", "VP", "SEQ_LEN", "PRECISION", "TOTAL_TOKENS", "MFU",
              "N_EXPERTS", "TOPK", "EXPERTS_PER_GPU")}
    try:
        return _estimate_training(config, variant, pc, ac, utils.core_calculations, utils.phase1_memory)
    finally:
        for k, v in saved.items():
            setattr(pc, k, v)
        importlib.reload(ac)
        importlib.reload(utils.core_calculations)
        importlib.reload(utils.phase1_memory)


def _estimate_training(config, variant, pc, ac, core, phase1_mod):
    precision = str(config.get("precision", "bf16")).lower()
    hardware_type = config["hardware"]
    num_gpus = int(config["num_gpus"])
    gpus_per_node = int(config.get("gpus_per_node") or 8)
    tp = int(config.get("tensor_parallel", 1) or 1)
    pp = int(config.get("pipeline_parallel", 1) or 1)
    ep = int(config.get("expert_parallel", 1) or 1)
    vp = int(config.get("virtual_pipeline_parallel", 1) or 1)
    sequence_length = int(config.get("sequence_length", 4096))
    micro_batch_size = config.get("micro_batch_size", 1)
    total_tokens_b = float(config.get("total_tokens_b", 100))
    variant = dict(variant)
    if variant.get("expert_ffn", 0) > 0:  # a variant without its own counts uses the project defaults
        variant.setdefault("num_experts", pc.N_EXPERTS)
        variant.setdefault("topk", pc.TOPK)
    n_experts = variant.get("num_experts", 0)
    topk = variant.get("topk", 0)
    num_nodes = max(1, math.ceil(num_gpus / gpus_per_node))  # a partial node is still paid for

    saved_vocab = pc.VOCAB
    pc.VOCAB = variant.get("vocab") or saved_vocab
    pc.TP, pc.PP, pc.EP, pc.CP, pc.VP = tp, pp, ep, 1, vp
    pc.SEQ_LEN = sequence_length
    pc.PRECISION = precision.upper()
    pc.TOTAL_TOKENS = total_tokens_b * 1e9
    pc.MFU = float(config.get("mfu") or 0.4)  # read only by the reference phases; not used for time
    pc.N_EXPERTS = n_experts
    pc.TOPK = topk
    pc.EXPERTS_PER_GPU = n_experts // ep if ep > 0 and n_experts > 0 else 0
    importlib.reload(ac)
    importlib.reload(core)
    importlib.reload(phase1_mod)

    if variant.get("expert_ffn", 0) == 0:  # dense model: no experts
        pc.N_EXPERTS = pc.TOPK = pc.EXPERTS_PER_GPU = 0
        pc.EP = ep = 1
        n_experts = topk = 0

    hw_specs = instance_specs(hardware_type)
    gpu_mem = hw_specs["mem_gb"]
    hw = {
        "name": f"{num_nodes} {hardware_type}", "gpus": num_gpus, "nodes": num_nodes, "mem_gb": gpu_mem,
        "label": f"{num_gpus}_{hardware_type}", "inter_node_bw_gb": hw_specs["inter_node_bw_gb"],
        "intra_node_bw_gbps": hw_specs["intra_node_bw_gbps"], "peak_tflops_bf16": hw_specs["peak_tflops_bf16"],
    }
    shape = variant.get("_shape") or shape_from_preset(variant, saved_vocab)
    mm_hw = mm.HARDWARE.get(hardware_type) or mm.HARDWARE["p5.48xlarge"]
    given_mfu = config.get("mfu")  # explicit override only; otherwise MFU is estimated

    def memory_for(try_tp, try_pp, try_ep):
        """Phase 1 memory with THIS TP/PP/EP (the modules read them at import, so reload).
        dp is Megatron's N / (TP x PP); EP divides it."""
        pc.TP, pc.PP, pc.EP = try_tp, try_pp, try_ep
        pc.EXPERTS_PER_GPU = n_experts // try_ep if n_experts else 0
        importlib.reload(core)
        importlib.reload(phase1_mod)
        return phase1_mod.analyze_variant_memory(
            variant, hw, num_gpus // (try_tp * try_pp), math.ceil(variant["layers"] / try_pp),
            pc.EXPERTS_PER_GPU)

    def plan_estimate(c, p1, mbs):
        plan = {"num_gpus": num_gpus, "tp": c["tp"], "pp": c["pp"], "cp": 1, "vp": vp, "ep": c["ep"],
                "mbs": mbs, "seq": sequence_length, "gbs": config.get("global_batch_size"),
                "zero": max(1, p1["best_zero"]) if p1 else 1}
        return mm.estimate(shape, mm_hw, plan, precision)

    # Plan search: candidates of mfu_model.candidate_plans, restricted to what the request fixed,
    # ranked by estimated throughput; the first ones that fit in memory are scored with each
    # viable micro batch and the fastest wins, so bubble and exposed communication count.
    fixed = {k: v for k, v in (("tp", tp), ("pp", pp), ("ep", ep)) if v and v != 1}
    auto_selected = not fixed
    cands = [c for c in mm.candidate_plans(num_gpus, variant["layers"], n_experts, gpus_per_node, sequence_length)
             if all(c[k] == v for k, v in fixed.items())]
    if not cands:
        return {"error": f"No valid plan with {', '.join(f'{k.upper()}={v}' for k, v in fixed.items())} on "
                         f"{num_gpus} GPUs: TP x PP must divide the GPU count, TP <= GPUs per node, EP must "
                         "divide the experts and the GPUs per pipeline stage."}
    ranked = sorted(cands, key=lambda c: -plan_estimate(c, None, 1)["tokens_per_second"])
    max_fitting = int(config.get("max_fitting_plans") or 6)  # plans that fit memory to score (speed cap)
    best, tried_plans, fitting = None, 0, 0
    for c in ranked:
        if fitting >= max_fitting or tried_plans >= 60:
            break
        tried_plans += 1
        p1 = memory_for(c["tp"], c["pp"], c["ep"])
        if p1["best_zero"] == -1:
            continue
        fitting += 1
        micros = p1.get(f"viable_micros_z{p1['best_zero']}") or [p1["best_micro"] or 1]
        for mbs in sorted(set(int(x) for x in micros if x))[:6]:
            est = plan_estimate(c, p1, mbs)
            if best is None or est["tokens_per_second"] > best[2]["tokens_per_second"]:
                best = (c, p1, est, mbs)
    if best is None:  # nothing fits: report the top-ranked plan as OOM
        c = ranked[0]
        p1 = memory_for(c["tp"], c["pp"], c["ep"])
        best = (c, p1, plan_estimate(c, p1, 1), 1)
    choice, p1, est, chosen_mbs = best
    tp, pp, ep = choice["tp"], choice["pp"], choice["ep"]
    dp = num_gpus // (tp * pp)
    layers_per_gpu = math.ceil(variant["layers"] / pp)

    # Throughput, time and cost from the estimated MFU.
    if given_mfu:
        mfu_value = float(given_mfu)
        tokens_per_second = mfu_value * est["peak_tflops_per_gpu"] * 1e12 * num_gpus / est["flops_per_token"]
        mfu_source = "given in the request"
    else:
        mfu_value = est["mfu"]
        tokens_per_second = est["tokens_per_second"]
        mfu_source = "estimated (mfu_model)"
    wall_tokens_per_second = tokens_per_second * mm.GOODPUT
    training_hours = total_tokens_b * 1e9 / wall_tokens_per_second / 3600
    training_days = training_hours / 24
    cost_per_hour = instance_cost(hardware_type) * num_nodes
    total_cost = cost_per_hour * training_hours

    is_oom = p1["best_zero"] == -1
    memory_breakdown = p1.get("memory_breakdown")
    t = est["time_s"]
    result = {
        "calculation_type": "Calculated/Projected",
        "disclaimer": "Reference ai-model-calculator memory model; MFU, step time and throughput from "
                      f"mfu_model ({mm.MODEL_VERSION}), see docs/CALCULATOR-MFU.md.",
        "model": {
            "name": variant["name"],
            "total_params_B": variant["total_params_B"],
            "active_params_B": variant.get("active_params_B", variant["total_params_B"]),
            "layers": variant["layers"],
            "hidden_dim": variant["d"],
            **({"attention": variant["attention"]} if variant.get("attention") else {}),
        },
        "hardware": {
            "instance": hardware_type,
            "gpu": GPU_OF_INSTANCE.get(hardware_type, ""),
            "gpus_per_node": gpus_per_node,
            "peak_tflops_bf16_per_gpu": mm_hw["bf16"],
            "peak_tflops_fp8_per_gpu": mm_hw["fp8"],
            "nvlink_gb_s_per_gpu": mm_hw["nvlink"],
            "network_gb_s_per_gpu": mm_hw["net"],
            "price_per_instance_hour_usd": instance_cost(hardware_type),
        },
        "inputs": {
            "precision": precision.upper(),
            "sequence_length": sequence_length,
            "micro_batch_size": micro_batch_size,
            "global_batch_size": config.get("global_batch_size"),
            "total_tokens_b": total_tokens_b,
            "mfu_source": mfu_source,
        },
        "fits_in_memory": not is_oom,
        "parallelism": {
            "tensor_parallel": tp,
            "pipeline_parallel": pp,
            "data_parallel": dp,
            "expert_parallel": ep,
            "auto_selected": auto_selected,
            "zero_stage": p1["best_zero"],
            "micro_batch_size": chosen_mbs,
            "viable_micros_z1": p1["viable_micros_z1"],
            "viable_micros_z2": p1["viable_micros_z2"],
            "total_gpus": num_gpus,
            "num_nodes": num_nodes,
            "layers_per_gpu": layers_per_gpu,
            "global_batch_sequences": est["plan"]["gbs"],
            "global_batch_tokens": est["plan"]["tokens_per_step"],
            "global_batch_source": ("request" if config.get("global_batch_size") else
                                    "default 16M tokens per step (Llama 3 405B, Llama 3 paper Table 4)"),
            "micro_batches_per_step": est["plan"]["micro_batches"],
            "plans_checked": tried_plans,
            "convention": "Megatron: GPUs = TP x PP x DP; EP is carved out of DP",
        },
        "training_time": {
            "total_tokens_b": total_tokens_b,
            "tokens_per_second": int(tokens_per_second),
            "tokens_per_second_wall_clock": int(wall_tokens_per_second),
            "goodput": mm.GOODPUT,
            "training_hours": round(training_hours, 1),
            "training_days": round(training_days, 1),
            "training_months": round(training_days / 30.44, 2),
            "mfu": round(mfu_value, 4),
            "model_tflops_per_gpu": round(mfu_value * est["peak_tflops_per_gpu"], 1),
        },
        "cost_estimate": {
            "cost_per_hour_usd": round(cost_per_hour, 2),
            "total_cost_usd": round(total_cost, 0),
            "note": "On-demand pricing estimate",
        },
        "mfu_breakdown": {
            "model_version": mm.MODEL_VERSION,
            "precision_peak_tflops_per_gpu": est["peak_tflops_per_gpu"],
            "flops_per_token": round(est["flops_per_token"]),
            "attention_flops_share": est["attention_flops_share"],
            "expert_flops_share": est["expert_flops_share"],
            "base_efficiency": est["base_efficiency"],
            "base_efficiency_key": f"{mm_hw['family']}/{est['precision']}",
            "moe_expert_factor": est["moe_expert_factor"],
            "compute_efficiency": est["compute_efficiency"],
            "bubble_fraction": est["bubble_fraction"],
            "step_time_s": round(t["step"], 4),
            "compute_s": round(t["compute"], 4),
            "bubble_s": round(t["bubble"], 4),
            "tp_exposed_s": round(t["tp_exposed"], 4),
            "ep_exposed_s": round(t["ep_exposed"], 4),
            "dp_exposed_s": round(t["dp_exposed"], 4),
            "tp_total_s": round(t["tp_total"], 4),
            "ep_total_s": round(t["ep_total"], 4),
            "dp_total_s": round(t["dp_total"], 4),
            "overlap_assumed": dict(mm.OVERLAP),
            "link_efficiency_assumed": dict(mm.LINK_EFFICIENCY),
            "goodput": mm.GOODPUT,
            "mfu_estimated": round(est["mfu"], 4),
            "validation": dict(mm.VALIDATION_SUMMARY),
            "sources": dict(mm.SOURCES),
        },
    }
    if memory_breakdown:
        result["memory_breakdown"] = {
            "model_params_gb": round(memory_breakdown["model"], 2),
            "gradients_gb": round(memory_breakdown["grad"], 2),
            "optimizer_states_gb": round(memory_breakdown["optim"], 2),
            "activations_gb": round(memory_breakdown["activation"], 2),
            "buffers_gb": round(memory_breakdown["buffer"], 2),
            "total_per_gpu_gb": round(memory_breakdown["total"], 2),
            "gpu_capacity_gb": gpu_mem,
            "headroom_gb": round(gpu_mem - memory_breakdown["total"], 2),
            "utilization_pct": round((memory_breakdown["total"] / gpu_mem) * 100, 1),
        }
    elif p1.get("oom_diagnostics"):
        diag = p1["oom_diagnostics"]
        result["memory_breakdown"] = {
            "model_params_gb": round(p1.get("model_mem_gb", 0), 2),
            "total_per_gpu_gb": round(diag["required_memory"], 2),
            "gpu_capacity_gb": diag["available_memory"],
            "headroom_gb": round(-diag["shortage"], 2),
            "utilization_pct": round((diag["required_memory"] / diag["available_memory"]) * 100, 1),
        }
        result["oom_diagnostics"] = {"shortage_gb": round(diag["shortage"], 2),
                                     "suggestions": diag.get("suggestions", [])}
    result["communication_overhead"] = {
        "model": mm.MODEL_VERSION,
        "dp_grad_reduce_and_param_gather_ms": round(t["dp_total"] * 1e3, 2),
        "dp_exposed_ms": round(t["dp_exposed"] * 1e3, 2),
        "tp_ms": round(t["tp_total"] * 1e3, 2),
        "tp_exposed_ms": round(t["tp_exposed"] * 1e3, 2),
        "ep_all_to_all_ms": round(t["ep_total"] * 1e3, 2),
        "ep_exposed_ms": round(t["ep_exposed"] * 1e3, 2),
        "selected_zero_stage": p1["best_zero"],
        "data_parallel_degree": dp,
        "network_gb_s_per_gpu": mm_hw["net"],
        "nvlink_gb_s_per_gpu": mm_hw["nvlink"],
    }
    if pp > 1:
        result["pipeline_communication"] = {
            "pipeline_bubble_pct": round(est["bubble_fraction"] * 100, 2),
            "num_microbatches": est["plan"]["micro_batches"],
            "virtual_pipeline": vp,
            "formula": "(PP - 1) / (VP x m + PP - 1)",
            "bubble_ms": round(t["bubble"] * 1e3, 2),
        }
    if ep > 1 and n_experts:
        result["alltoall_communication"] = {
            "a2a_per_step_ms": round(t["ep_total"] * 1e3, 2),
            "a2a_exposed_ms": round(t["ep_exposed"] * 1e3, 2),
            "comm_type": ("intra-node (NVLink)" if ep * est["plan"]["etp"] <= gpus_per_node
                          else "inter-node (EFA)"),
        }
    return result


def run_estimate(config, variant):
    """estimate_training for a given GPU count, or the smallest whole-node GPU count (doubling) that
    fits in memory and meets config["target_days"] when num_gpus is None / "auto"."""
    config = dict(config or {})
    config["hardware"] = normalise_hardware(config.get("hardware") or "p5.48xlarge")
    gpus_per_node = int(config.get("gpus_per_node") or 8)
    num_gpus = config.get("num_gpus")
    if num_gpus in (None, "", 0, "auto"):
        target_days = config.get("target_days")
        tried, n, result = [], gpus_per_node, None
        while n <= MAX_AUTO_GPUS:
            result = estimate_training({**config, "num_gpus": n}, variant)
            if "error" in result:
                return result
            days = result["training_time"]["training_days"]
            tried.append({"num_gpus": n, "fits": result["fits_in_memory"], "training_days": days})
            if result["fits_in_memory"] and (not target_days or days <= float(target_days)):
                break
            n *= 2
        result["gpu_count_search"] = {
            "basis": f"smallest GPU count (whole {gpus_per_node}-GPU nodes, doubling) that fits"
                     + (f" and trains in <= {target_days} days" if target_days else " in memory"),
            "tried": tried,
        }
        return result
    return estimate_training({**config, "num_gpus": int(num_gpus)}, variant)


def format_estimate(r):
    """Concise markdown summary of an estimate_training result."""
    if "error" in r:
        return f"Error: {r['error']}"
    p, t, c, b = r["parallelism"], r["training_time"], r["cost_estimate"], r["mfu_breakdown"]
    lines = [
        f"## {r['model']['name']} on {p['total_gpus']} x {r['hardware']['gpu'] or r['hardware']['instance']}",
        "",
        f"- **Plan**: TP={p['tensor_parallel']} PP={p['pipeline_parallel']} EP={p['expert_parallel']} "
        f"DP={p['data_parallel']}, micro batch {p['micro_batch_size']}, ZeRO-{p['zero_stage']}, "
        f"{p['micro_batches_per_step']} micro-batches/step, {p['global_batch_tokens'] / 1e6:.1f}M tokens/step "
        f"({p['global_batch_source']})",
        f"- **Fits in memory**: {'yes' if r['fits_in_memory'] else 'NO (out of memory)'}",
        f"- **MFU**: {t['mfu'] * 100:.1f}% ({r['inputs']['mfu_source']}), {t['model_tflops_per_gpu']} TFLOPS/GPU",
        f"- **Throughput**: {t['tokens_per_second']:,} tokens/s ({t['tokens_per_second_wall_clock']:,} wall clock "
        f"at {t['goodput'] * 100:.0f}% goodput)",
        f"- **Time**: {t['training_days']:,.1f} days for {t['total_tokens_b']:,.0f}B tokens",
        f"- **Cost**: ${c['total_cost_usd']:,.0f} (${c['cost_per_hour_usd']:,.2f}/h, {c['note']})",
        f"- **Bubble**: {b['bubble_fraction'] * 100:.1f}%, step {b['step_time_s']} s "
        f"(compute {b['compute_s']}, bubble {b['bubble_s']}, exposed TP {b['tp_exposed_s']} "
        f"EP {b['ep_exposed_s']} DP {b['dp_exposed_s']})",
        f"- **Validation**: median error {b['validation']['median_abs_error_pct']}% over "
        f"{b['validation']['runs']} published runs (docs/CALCULATOR-MFU.md)",
    ]
    return "\n".join(lines)
