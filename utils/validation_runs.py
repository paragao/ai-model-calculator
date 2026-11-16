"""
Validation set for mfu_model: real, published training runs across model families.

Each run: architecture (Hugging Face config.json field names), hardware key (mfu_model.HARDWARE),
the plan as published, the reported number and its source. `fit=True` runs are the only ones
used to fit mfu_model.BASE_EFFICIENCY and MOE_EXPERT_FACTOR; all others are hold-out.

    python3 validation_runs.py          # table of predicted vs reported, per run and overall
    python3 validation_runs.py --fit    # refit the constants from the fit runs and print them

Metrics: "tokens_per_gpu" (tokens per second per GPU; clean across FLOP conventions),
"tokens_per_gpu_wallclock" (derived from published GPU-hours; compared x mfu_model.GOODPUT), or
"mfu" (when only TFLOPS/MFU is published; compared with this model's own FLOP count).
"""
from __future__ import annotations

import json
import math
import statistics
import sys

from utils import mfu_model as mm

# --- Architectures (config.json fields of the public checkpoints, or the paper's table) ---------
ARCH = {
    "Llama-3-8B": {"num_hidden_layers": 32, "hidden_size": 4096, "intermediate_size": 14336,
                   "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 128256,
                   "tie_word_embeddings": False, "hidden_act": "silu"},
    "Llama-3-70B": {"num_hidden_layers": 80, "hidden_size": 8192, "intermediate_size": 28672,
                    "num_attention_heads": 64, "num_key_value_heads": 8, "vocab_size": 128256,
                    "tie_word_embeddings": False, "hidden_act": "silu"},
    "Llama-3.1-405B": {"num_hidden_layers": 126, "hidden_size": 16384, "intermediate_size": 53248,
                       "num_attention_heads": 128, "num_key_value_heads": 8, "vocab_size": 128256,
                       "tie_word_embeddings": False, "hidden_act": "silu"},
    "Mixtral-8x7B": {"num_hidden_layers": 32, "hidden_size": 4096, "intermediate_size": 14336,
                     "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 32000,
                     "num_local_experts": 8, "num_experts_per_tok": 2, "tie_word_embeddings": False,
                     "hidden_act": "silu"},
    "Mixtral-8x22B": {"num_hidden_layers": 56, "hidden_size": 6144, "intermediate_size": 16384,
                      "num_attention_heads": 48, "num_key_value_heads": 8, "vocab_size": 32768,
                      "num_local_experts": 8, "num_experts_per_tok": 2, "tie_word_embeddings": False,
                      "hidden_act": "silu"},
    "DeepSeek-V3": {"num_hidden_layers": 61, "hidden_size": 7168, "intermediate_size": 18432,
                    "moe_intermediate_size": 2048, "num_attention_heads": 128, "num_key_value_heads": 128,
                    "q_lora_rank": 1536, "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64,
                    "v_head_dim": 128, "n_routed_experts": 256, "n_shared_experts": 1, "num_experts_per_tok": 8,
                    "first_k_dense_replace": 3, "vocab_size": 129280, "tie_word_embeddings": False,
                    "hidden_act": "silu", "n_group": 8, "topk_group": 4, "num_nextn_predict_layers": 1},
    "GPT-3-175B": {"num_hidden_layers": 96, "hidden_size": 12288, "intermediate_size": 49152,
                   "num_attention_heads": 96, "vocab_size": 50257, "tie_word_embeddings": True,
                   "activation_function": "gelu"},
    "Nemotron-4-340B": {"num_hidden_layers": 96, "hidden_size": 18432, "intermediate_size": 73728,
                        "num_attention_heads": 96, "num_key_value_heads": 8, "vocab_size": 256000,
                        "tie_word_embeddings": False, "hidden_act": "relu2"},
    "MT-NLG-530B": {"num_hidden_layers": 105, "hidden_size": 20480, "intermediate_size": 81920,
                    "num_attention_heads": 128, "vocab_size": 51200, "tie_word_embeddings": True,
                    "activation_function": "gelu"},
    "Qwen3-8B": {"num_hidden_layers": 36, "hidden_size": 4096, "intermediate_size": 12288,
                 "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936,
                 "tie_word_embeddings": False, "hidden_act": "silu"},
    "Qwen3-30B-A3B": {"num_hidden_layers": 48, "hidden_size": 2048, "intermediate_size": 6144,
                      "moe_intermediate_size": 768, "num_attention_heads": 32, "num_key_value_heads": 4,
                      "head_dim": 128, "num_experts": 128, "num_experts_per_tok": 8, "vocab_size": 151936,
                      "tie_word_embeddings": False, "hidden_act": "silu"},
}
# Published parameter counts (billions): total, active. Checks the generic config parser.
PUBLISHED_PARAMS = {
    "Llama-3-8B": (8.03, 8.03), "Llama-3-70B": (70.6, 70.6), "Llama-3.1-405B": (405.9, 405.9),
    "Mixtral-8x7B": (46.7, 12.9), "Mixtral-8x22B": (141.0, 39.0), "DeepSeek-V3": (671.0, 37.0),
    "GPT-3-175B": (175.0, 175.0), "Nemotron-4-340B": (341.0, 341.0), "MT-NLG-530B": (530.0, 530.0),
    "Qwen3-8B": (8.2, 8.2), "Qwen3-30B-A3B": (30.5, 3.3),
}

NEMO_2507 = "https://docs.nvidia.com/nemo-framework/user-guide/25.11/performance/archive/25.07-performance-summary.html"
NEMO_2504 = "https://docs.nvidia.com/nemo-framework/user-guide/25.07/performance/archive/25.04-performance-summary.html"
LLAMA3 = "https://arxiv.org/abs/2407.21783 (Table 4)"
DSV3 = "https://arxiv.org/abs/2412.19437 (Sec. 1: 180K H800 GPU hours per trillion tokens; Sec. 3.2, 4.2)"
MTNLG = "https://arxiv.org/abs/2201.11990 (Sec. 2.4: 126/121/113 TFLOP/s per GPU)"
REPO = "repos/benchmarks/"


def _run(rid, model, hw, prec, n, gbs, mbs, seq, tp=1, pp=1, cp=1, vp=1, ep=1, fsdp=False, *,
         metric, value, source, fit=False, note="", overlap_ep=None, dedup_by_group=False):
    plan = {"num_gpus": n, "gbs": gbs, "mbs": mbs, "seq": seq, "tp": tp, "pp": pp, "cp": cp,
            "vp": vp, "ep": ep, "fsdp": fsdp}
    if overlap_ep is not None:
        plan["overlap_ep"] = overlap_ep
    if dedup_by_group:
        plan["dedup_by_group"] = True
    return {"id": rid, "model": model, "hardware": hw, "precision": prec, "plan": plan,
            "metric": metric, "reported": value, "source": source, "fit": fit, "note": note}


RUNS = [
    # NeMo 25.07, DGX-H100, FP8
    _run("nemo-h100-llama3-8b", "Llama-3-8B", "dgx-h100", "fp8", 8, 128, 1, 8192, fsdp=True,
         metric="tokens_per_gpu", value=14744, source=NEMO_2507, fit=True),
    _run("nemo-h100-llama3-70b", "Llama-3-70B", "dgx-h100", "fp8", 64, 128, 1, 8192, tp=4, pp=8, vp=5,
         metric="tokens_per_gpu", value=1663, source=NEMO_2507, fit=True),
    _run("nemo-h100-llama31-405b", "Llama-3.1-405B", "dgx-h100", "fp8", 1024, 512, 1, 8192, tp=8, pp=8, cp=2, vp=8,
         metric="tokens_per_gpu", value=320, source=NEMO_2507),
    _run("nemo-h100-mixtral-8x7b", "Mixtral-8x7B", "dgx-h100", "fp8", 64, 256, 1, 4096, pp=4, vp=8, ep=8,
         metric="tokens_per_gpu", value=8275, source=NEMO_2507, fit=True),
    _run("nemo-h100-mixtral-8x22b", "Mixtral-8x22B", "dgx-h100", "fp8", 256, 256, 1, 65536, tp=4, pp=4, cp=8, vp=14, ep=8,
         metric="tokens_per_gpu", value=1280, source=NEMO_2507),
    _run("nemo-h100-deepseek-v3", "DeepSeek-V3", "dgx-h100", "fp8", 1024, 8192, 1, 4096, tp=2, pp=16, vp=2, ep=64,
         metric="tokens_per_gpu", value=887, source=NEMO_2507, fit=True),
    _run("nemo-h100-gpt3-175b", "GPT-3-175B", "dgx-h100", "fp8", 512, 2048, 2, 2048, tp=4, pp=8, vp=6,
         metric="tokens_per_gpu", value=824, source=NEMO_2504),
    _run("nemo-h100-nemotron4-340b", "Nemotron-4-340B", "dgx-h100", "fp8", 256, 64, 1, 4096, tp=8, pp=8, vp=12,
         metric="tokens_per_gpu", value=335, source=NEMO_2504),
    # NeMo 25.07, DGX-B200, FP8
    _run("nemo-b200-llama3-8b", "Llama-3-8B", "dgx-b200", "fp8", 8, 128, 2, 8192,
         metric="tokens_per_gpu", value=30131, source=NEMO_2507, fit=True),
    _run("nemo-b200-llama3-70b", "Llama-3-70B", "dgx-b200", "fp8", 64, 128, 1, 8192, fsdp=True,
         metric="tokens_per_gpu", value=3690, source=NEMO_2507, fit=True),
    _run("nemo-b200-llama31-405b", "Llama-3.1-405B", "dgx-b200", "fp8", 128, 64, 1, 8192, tp=4, pp=8, cp=2, vp=8,
         metric="tokens_per_gpu", value=674, source=NEMO_2507),
    _run("nemo-b200-mixtral-8x7b", "Mixtral-8x7B", "dgx-b200", "fp8", 64, 256, 2, 4096, ep=8,
         metric="tokens_per_gpu", value=16384, source=NEMO_2507),
    _run("nemo-b200-mixtral-8x22b", "Mixtral-8x22B", "dgx-b200", "fp8", 256, 64, 1, 65536, tp=2, pp=4, cp=8, vp=14, ep=8,
         metric="tokens_per_gpu", value=2548, source=NEMO_2507),
    _run("nemo-b200-deepseek-v3", "DeepSeek-V3", "dgx-b200", "fp8", 256, 2048, 1, 4096, tp=2, pp=16, ep=8,
         metric="tokens_per_gpu", value=1640, source=NEMO_2507),
    _run("nemo-b200-gpt3-175b", "GPT-3-175B", "dgx-b200", "fp8", 512, 2048, 2, 2048, tp=4, pp=4, vp=6,
         metric="tokens_per_gpu", value=1523, source=NEMO_2504),
    _run("nemo-b200-nemotron4-340b", "Nemotron-4-340B", "dgx-b200", "fp8", 128, 32, 1, 4096, tp=8, pp=4, vp=12,
         metric="tokens_per_gpu", value=632, source=NEMO_2504),
    # Meta, Llama 3 405B pre-training, BF16 (VP and micro batch not published: 1 assumed)
    _run("meta-llama3-405b-8k-gpus", "Llama-3.1-405B", "meta-h100-roce", "bf16", 8192, 2048, 1, 8192, tp=8, pp=16, vp=8,
         metric="mfu", value=0.43, source=LLAMA3,
         note="batch/DP 32 x DP 64; VP and MBS not published: VP 8 and MBS 1 as in the NeMo Llama3.1-405B recipe"),
    _run("meta-llama3-405b-16k-gpus", "Llama-3.1-405B", "meta-h100-roce", "bf16", 16384, 2048, 1, 8192, tp=8, pp=16, vp=8,
         metric="mfu", value=0.41, source=LLAMA3,
         note="batch/DP 16 x DP 128; VP and MBS not published: VP 8 and MBS 1 as in the NeMo Llama3.1-405B recipe"),
    # DeepSeek-V3 pre-training, FP8, 2048 H800: 1e12 tokens / (180e3 GPU-h x 3600 s) per GPU
    _run("deepseek-v3-paper", "DeepSeek-V3", "deepseek-h800", "fp8", 2048, 15360, 1, 4096, pp=16, ep=64, vp=2,
         metric="tokens_per_gpu_wallclock", value=round(1e12 / (180e3 * 3600), 1), source=DSV3, overlap_ep=0.95, dedup_by_group=True,
         note="final batch 15360 x 4096; DualPipe: bubble about half of 1F1B (VP 2) and all-to-all "
              "nearly fully overlapped (paper Sec. 3.2.1), so EP overlap 0.95; node-limited routing "
              "(at most 4 nodes per token) with per-node de-duplication (Sec. 3.2.2)"),
    # Megatron-Turing NLG 530B, A100 40GB, mixed precision, TP8 PP35
    _run("mtnlg-530b-2240-gpus", "MT-NLG-530B", "dgx-a100-40g", "bf16", 2240, 1920, 1, 2048, tp=8, pp=35,
         metric="mfu", value=round(126 / 312, 4), source=MTNLG, fit=True),
    _run("mtnlg-530b-2800-gpus", "MT-NLG-530B", "dgx-a100-40g", "bf16", 2800, 1920, 1, 2048, tp=8, pp=35,
         metric="mfu", value=round(121 / 312, 4), source=MTNLG),
    _run("mtnlg-530b-3360-gpus", "MT-NLG-530B", "dgx-a100-40g", "bf16", 3360, 1920, 1, 2048, tp=8, pp=35,
         metric="mfu", value=round(113 / 312, 4), source=MTNLG),
    # Our own pre-training runs (AWS)
    _run("repo-llama70b-bf16-p5en", "Llama-3-70B", "p5en.48xlarge", "bf16", 64, 128, 1, 8192, tp=2, pp=4, cp=2, vp=5,
         metric="tokens_per_gpu", value=round(76819 / 64, 1), source=REPO + "pretrain-llama70b-bf16/p5en-8nodes", fit=True),
    _run("repo-llama70b-fp8-p5en", "Llama-3-70B", "p5en.48xlarge", "fp8", 64, 128, 1, 8192, tp=4, pp=2, vp=5,
         metric="tokens_per_gpu", value=round(117162 / 64, 1), source=REPO + "pretrain-llama70b-fp8/p5en-8nodes"),
    _run("repo-llama70b-bf16-b200", "Llama-3-70B", "p6-b200.48xlarge", "bf16", 64, 128, 2, 8192, tp=2, pp=4, cp=2, vp=5,
         metric="tokens_per_gpu", value=round(119066 / 64, 1), source=REPO + "pretrain-llama70b-bf16/p6-b200-8nodes", fit=True),
    _run("repo-llama70b-fp8-b200", "Llama-3-70B", "p6-b200.48xlarge", "fp8", 64, 128, 1, 8192, fsdp=True,
         metric="tokens_per_gpu", value=round(222354 / 64, 1), source=REPO + "pretrain-llama70b-fp8/p6-b200-8nodes",
         note="FSDP with CPU offload (offload not modelled)"),
    _run("repo-qwen3-8b-bf16-p5en", "Qwen3-8B", "p5en.48xlarge", "bf16", 16, 128, 2, 4096,
         metric="tokens_per_gpu", value=round(162000 / 16, 1), source=REPO + "pretrain-qwen3-8b-bf16/p5en-2nodes", fit=True),
    _run("repo-qwen3-8b-bf16-b300", "Qwen3-8B", "p6-b300.48xlarge", "bf16", 16, 128, 4, 4096,
         metric="tokens_per_gpu", value=round(318000 / 16, 1), source=REPO + "pretrain-qwen3-8b-bf16/p6-b300-2nodes", fit=True),
    _run("repo-qwen3-30b-a3b-bf16-p5en", "Qwen3-30B-A3B", "p5en.48xlarge", "bf16", 64, 2048, 2, 4096, ep=8,
         metric="mfu", value=0.337, source=REPO + "pretrain-qwen3-30b-moe-bf16/p5en-8nodes", fit=True),
]


def predict(run, shapes=None):
    shapes = shapes if shapes is not None else {}
    if run["model"] not in shapes:
        shapes[run["model"]] = mm.model_shape(ARCH[run["model"]])
    est = mm.estimate(shapes[run["model"]], mm.HARDWARE[run["hardware"]], run["plan"], run["precision"])
    if run["metric"] == "tokens_per_gpu":
        pred = est["tokens_per_second_per_gpu"]
    elif run["metric"] == "tokens_per_gpu_wallclock":  # from GPU-hours: includes downtime
        pred = est["tokens_per_second_per_gpu"] * mm.GOODPUT
    else:
        pred = est["mfu"]
    return pred, est


def evaluate(runs=RUNS):
    rows, shapes = [], {}
    for r in runs:
        pred, est = predict(r, shapes)
        rows.append({"id": r["id"], "model": r["model"], "hardware": r["hardware"], "precision": r["precision"],
                     "gpus": r["plan"]["num_gpus"], "metric": r["metric"], "reported": r["reported"],
                     "predicted": round(pred, 4 if r["metric"] == "mfu" else 1),
                     "predicted_mfu": round(est["mfu"], 4), "error_pct": round((pred / r["reported"] - 1) * 100, 1),
                     "fit": r["fit"], "source": r["source"], "note": r["note"]})
    return rows


def summary(rows):
    def stats(sel):
        errs = [abs(x["error_pct"]) for x in sel]
        return {"runs": len(errs), "median_abs_error_pct": round(statistics.median(errs), 1) if errs else None,
                "max_abs_error_pct": round(max(errs), 1) if errs else None}
    return {"all": stats(rows), "hold_out": stats([x for x in rows if not x["fit"]]),
            "fit": stats([x for x in rows if x["fit"]])}


def _solve(fn, target, lo=0.05, hi=1.5):
    """x with fn(x) == target (fn increasing), by bisection."""
    for _ in range(60):
        mid = (lo + hi) / 2
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fit():
    """Fit BASE_EFFICIENCY per (family, precision) on dense fit runs, then MOE_EXPERT_FACTOR on MoE fit runs.
    Each cell is the geometric mean of the per-run solutions."""
    saved_base, saved_moe, saved_half = dict(mm.BASE_EFFICIENCY), mm.MOE_EXPERT_FACTOR, mm.MOE_EXPERT_HALF_FFN
    shapes = {}
    cells = {}
    for r in RUNS:
        if not r["fit"]:
            continue
        shape = shapes.setdefault(r["model"], mm.model_shape(ARCH[r["model"]]))
        if shape["n_experts"]:
            continue
        key = (mm.HARDWARE[r["hardware"]]["family"], r["precision"])

        def f(x, r=r, key=key):
            mm.BASE_EFFICIENCY[key] = x
            return predict(r, shapes)[0]
        cells.setdefault(key, []).append(_solve(f, r["reported"]))
    fitted = {k: round(math.exp(statistics.mean(math.log(v) for v in vs)), 3) for k, vs in cells.items()}
    mm.BASE_EFFICIENCY.update(fitted)
    moe_runs = [r for r in RUNS if r["fit"] and shapes.setdefault(r["model"], mm.model_shape(ARCH[r["model"]]))["n_experts"]]
    best = None
    for fmax in [x / 100 for x in range(30, 101, 1)]:
        for half in (0, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096):
            mm.MOE_EXPERT_FACTOR, mm.MOE_EXPERT_HALF_FFN = fmax, half
            loss = sum(math.log(predict(r, shapes)[0] / r["reported"]) ** 2 for r in moe_runs)
            if best is None or loss < best[0]:
                best = (loss, fmax, half)
    moe_factor, moe_half = (best[1], best[2]) if best else (saved_moe, saved_half)
    mm.BASE_EFFICIENCY.clear()
    mm.BASE_EFFICIENCY.update(saved_base)
    mm.MOE_EXPERT_FACTOR, mm.MOE_EXPERT_HALF_FFN = saved_moe, saved_half
    return {"base_efficiency": {f"{k[0]}/{k[1]}": v for k, v in sorted(fitted.items())},
            "moe_expert_factor": moe_factor, "moe_expert_half_ffn": moe_half, "fit_runs": [r["id"] for r in RUNS if r["fit"]]}


def days_per_trillion(row, value):
    """Days to train 1T tokens on the run's GPUs at a tokens/s/GPU value (None for MFU-only runs)."""
    if row["metric"] == "mfu":
        return None
    return 1e12 / (value * row["gpus"]) / 86400


def markdown(rows):
    out = ["| Run | Model | GPUs | Hardware | Prec | Metric | Reported | Predicted | Error | "
           "Predicted MFU | Days per 1T tokens, reported / predicted | Set |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        fmt = (lambda v: f"{v * 100:.1f}%") if x["metric"] == "mfu" else (lambda v: f"{v:,.0f} tok/s/GPU")
        dr, dp = days_per_trillion(x, x["reported"]), days_per_trillion(x, x["predicted"])
        days = f"{dr:,.1f} / {dp:,.1f}" if dr else "–"
        out.append(f"| {x['id']} | {x['model']} | {x['gpus']} | {x['hardware']} | {x['precision']} | {x['metric']} | "
                   f"{fmt(x['reported'])} | {fmt(x['predicted'])} | {x['error_pct']:+.1f}% | "
                   f"{x['predicted_mfu'] * 100:.1f}% | {days} | {'fit' if x['fit'] else 'hold-out'} |")
    return "\n".join(out)


def html_table(rows):
    """One-line HTML table (for an MR description pushed as a git push option)."""
    cells = ["<tr><th>Run</th><th>GPUs</th><th>Hardware</th><th>Prec</th><th>Reported</th><th>Predicted</th>"
             "<th>Error</th><th>Pred. MFU</th><th>Days/1T rep. / pred.</th><th>Set</th></tr>"]
    for x in rows:
        fmt = (lambda v: f"{v * 100:.1f} pct MFU") if x["metric"] == "mfu" else (lambda v: f"{v:,.0f} tok/s/GPU")
        dr, dp = days_per_trillion(x, x["reported"]), days_per_trillion(x, x["predicted"])
        days = f"{dr:,.1f} / {dp:,.1f}" if dr else "-"
        cells.append(f"<tr><td>{x['id']}</td><td>{x['gpus']}</td><td>{x['hardware']}</td><td>{x['precision']}</td>"
                     f"<td>{fmt(x['reported'])}</td><td>{fmt(x['predicted'])}</td><td>{x['error_pct']:+.1f} pct</td>"
                     f"<td>{x['predicted_mfu'] * 100:.1f} pct</td><td>{days}</td><td>{'fit' if x['fit'] else 'hold-out'}</td></tr>")
    return "<table>" + "".join(cells) + "</table>"


if __name__ == "__main__":
    if "--fit" in sys.argv:
        print(json.dumps(fit(), indent=1))
        sys.exit(0)
    rows = evaluate()
    print(markdown(rows))
    print(json.dumps(summary(rows), indent=1))
