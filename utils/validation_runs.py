"""
Validation set for mfu_model: real, published training runs across model families.

Each run: architecture (Hugging Face config.json field names), hardware key (mfu_model.HARDWARE),
the plan as published, the reported number and its source. `fit=True` runs are the only ones
used to fit mfu_model.BASE_EFFICIENCY, MOE_EXPERT_FACTOR / MOE_EXPERT_HALF_FFN and OVERLAP["ep"];
all others are hold-out.

Every MoE run states its layout source in `layout`: "pinned: <recipe at a release tag>" when the
expert TP, dispatcher and recompute come from the recipe of the release that produced the number,
"unverified: ..." when they are assumed. `superseded=True` runs are kept and printed but neither
fitted nor scored: a newer release of the same software, on the same hardware and model, is much
faster (MoE kernels and dispatchers changed), so they no longer describe what the hardware does.

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
    # https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json
    "Qwen3-235B-A22B": {"num_hidden_layers": 94, "hidden_size": 4096, "intermediate_size": 12288,
                        "moe_intermediate_size": 1536, "num_attention_heads": 64, "num_key_value_heads": 4,
                        "head_dim": 128, "num_experts": 128, "num_experts_per_tok": 8, "vocab_size": 151936,
                        "tie_word_embeddings": False, "hidden_act": "silu"},
    # https://huggingface.co/moonshotai/Kimi-K2-Base/blob/main/config.json
    "Kimi-K2": {"num_hidden_layers": 61, "hidden_size": 7168, "intermediate_size": 18432,
                "moe_intermediate_size": 2048, "num_attention_heads": 64, "num_key_value_heads": 64,
                "q_lora_rank": 1536, "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64,
                "v_head_dim": 128, "n_routed_experts": 384, "n_shared_experts": 1, "num_experts_per_tok": 8,
                "first_k_dense_replace": 1, "vocab_size": 163840, "tie_word_embeddings": False,
                "hidden_act": "silu", "n_group": 1, "topk_group": 1},
    # https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct/blob/main/config.json (text_config, mirror of Meta's):
    # intermediate_size_mlp 16384 is the dense FFN, intermediate_size 8192 the expert and shared-expert FFN, MoE in every
    # layer, top-1 routing plus one shared expert (Llama 4 model card). Chunked attention (8192) equals the 8192 sequence.
    "Llama-4-Scout": {"num_hidden_layers": 48, "hidden_size": 5120, "intermediate_size": 16384,
                      "moe_intermediate_size": 8192, "num_attention_heads": 40, "num_key_value_heads": 8,
                      "head_dim": 128, "vocab_size": 202048, "tie_word_embeddings": False, "num_local_experts": 16,
                      "num_experts_per_tok": 1, "n_shared_experts": 1, "hidden_act": "silu"},
    # https://huggingface.co/unsloth/Llama-4-Maverick-17B-128E-Instruct/blob/main/config.json: as Scout, 128 experts,
    # MoE in every second layer (interleave_moe_layer_step 2), the others dense.
    "Llama-4-Maverick": {"num_hidden_layers": 48, "hidden_size": 5120, "intermediate_size": 16384,
                         "moe_intermediate_size": 8192, "num_attention_heads": 40, "num_key_value_heads": 8,
                         "head_dim": 128, "vocab_size": 202048, "tie_word_embeddings": False, "num_local_experts": 128,
                         "num_experts_per_tok": 1, "n_shared_experts": 1, "interleave_moe_layer_step": 2,
                         "hidden_act": "silu"},
    # https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json (sliding window 128 on alternate layers: not modelled)
    "GPT-OSS-120B": {"num_hidden_layers": 36, "hidden_size": 2880, "intermediate_size": 2880, "num_attention_heads": 64,
                     "num_key_value_heads": 8, "head_dim": 64, "vocab_size": 201088, "tie_word_embeddings": False,
                     "num_local_experts": 128, "num_experts_per_tok": 4, "hidden_act": "silu"},
}
# Published parameter counts (billions): total, active. Checks the generic config parser.
PUBLISHED_PARAMS = {
    "Llama-3-8B": (8.03, 8.03), "Llama-3-70B": (70.6, 70.6), "Llama-3.1-405B": (405.9, 405.9),
    "Mixtral-8x7B": (46.7, 12.9), "Mixtral-8x22B": (141.0, 39.0), "DeepSeek-V3": (671.0, 37.0),
    "GPT-3-175B": (175.0, 175.0), "Nemotron-4-340B": (341.0, 341.0), "MT-NLG-530B": (530.0, 530.0),
    "Qwen3-8B": (8.2, 8.2), "Qwen3-30B-A3B": (30.5, 3.3), "Qwen3-235B-A22B": (235.0, 22.0),
    "Kimi-K2": (1040.0, 32.0),
    "Llama-4-Scout": (109.0, 17.0), "Llama-4-Maverick": (400.0, 17.0),
    # gpt-oss model card (arXiv 2508.10925): 116.83B total, 5.13B active per token. That active count leaves out the
    # 0.58B input-embedding table, which model_shape's active count includes: 5.13 + 0.58 = 5.71.
    "GPT-OSS-120B": (116.83, 5.71),
}

NEMO_2507 = "https://docs.nvidia.com/nemo-framework/user-guide/25.11/performance/archive/25.07-performance-summary.html"
NEMO_2504 = "https://docs.nvidia.com/nemo-framework/user-guide/25.07/performance/archive/25.04-performance-summary.html"
LLAMA3 = "https://arxiv.org/abs/2407.21783 (Table 4)"
DSV3 = "https://arxiv.org/abs/2412.19437 (Sec. 1: 180K H800 GPU hours per trillion tokens; Sec. 3.2, 4.2)"
MTNLG = "https://arxiv.org/abs/2201.11990 (Sec. 2.4: 126/121/113 TFLOP/s per GPU)"
REPO = "repos/benchmarks/"
MB = "https://docs.nvidia.com/nemo/megatron-bridge/nightly/performance-summary-archive.html"
# Recipes of the release that published the 26.04.01 numbers (v0.4.2 carries "26.04.01_perf_summary").
MB_V042 = "https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/v0.4.2/scripts/performance/configs/"
NEMO_R240 = "https://github.com/NVIDIA/NeMo/blob/r2.4.0/scripts/performance/"


def _run(rid, model, hw, prec, n, gbs, mbs, seq, tp=1, pp=1, cp=1, vp=1, ep=1, fsdp=False, *,
         metric, value, source, fit=False, note="", overlap_ep=None, dedup_by_group=False, etp=None,
         layout="", superseded=False, dispatcher=None, a2a_overlap=None, recompute=None):
    """dispatcher / a2a_overlap / recompute: the recipe's MoE token dispatcher, whether it overlaps
    the all-to-all with compute, and its recompute_modules (recorded as published)."""
    plan = {"num_gpus": n, "gbs": gbs, "mbs": mbs, "seq": seq, "tp": tp, "pp": pp, "cp": cp,
            "vp": vp, "ep": ep, "fsdp": fsdp}
    if overlap_ep is not None:
        plan["overlap_ep"] = overlap_ep
    if dedup_by_group:
        plan["dedup_by_group"] = True
    if etp is not None:
        plan["etp"] = etp
    for k, v in (("dispatcher", dispatcher), ("a2a_overlap", a2a_overlap), ("recompute", recompute)):
        if v is not None:
            plan[k] = v
    return {"id": rid, "model": model, "hardware": hw, "precision": prec, "plan": plan,
            "metric": metric, "reported": value, "source": source, "fit": fit and not superseded,
            "note": note, "layout": layout, "superseded": superseded}


RUNS = [
    # NeMo 25.07, DGX-H100, FP8
    _run("nemo-h100-llama3-8b", "Llama-3-8B", "dgx-h100", "fp8", 8, 128, 1, 8192, fsdp=True,
         metric="tokens_per_gpu", value=14744, source=NEMO_2507, fit=True),
    _run("nemo-h100-llama3-70b", "Llama-3-70B", "dgx-h100", "fp8", 64, 128, 1, 8192, tp=4, pp=8, vp=5,
         metric="tokens_per_gpu", value=1663, source=NEMO_2507, fit=True),
    _run("nemo-h100-llama31-405b", "Llama-3.1-405B", "dgx-h100", "fp8", 1024, 512, 1, 8192, tp=8, pp=8, cp=2, vp=8,
         metric="tokens_per_gpu", value=320, source=NEMO_2507),
    _run("nemo-h100-mixtral-8x7b", "Mixtral-8x7B", "dgx-h100", "fp8", 64, 256, 1, 4096, pp=4, vp=8, ep=8, etp=1,
         layout="pinned: " + NEMO_R240 + "recommended_model_configs/model_configs_h100.csv (etp_size 1 or unset at TP 1)",
         metric="tokens_per_gpu", value=8275, source=NEMO_2507, fit=True),
    _run("nemo-h100-mixtral-8x22b", "Mixtral-8x22B", "dgx-h100", "fp8", 256, 256, 1, 65536, tp=4, pp=4, cp=8, vp=14, ep=8, etp=1,
         layout="pinned: " + NEMO_R240 + "recommended_model_configs/model_configs_h100.csv (etp_size 1 or unset at TP 1)",
         metric="tokens_per_gpu", value=1280, source=NEMO_2507),
    _run("nemo-h100-deepseek-v3", "DeepSeek-V3", "dgx-h100", "fp8", 1024, 8192, 1, 4096, tp=2, pp=16, vp=2, ep=64,
         etp=1, dedup_by_group=True, dispatcher="deepep", recompute=["mla_up_proj", "mlp", "moe"],
         metric="tokens_per_gpu", value=887, source=NEMO_2507, superseded=True,
         layout="pinned: " + NEMO_R240 + "recommended_model_configs/model_configs_h100.csv (etp_size 1, "
                "recompute mla_up_proj/mlp/moe) and llm/pretrain_deepseek_v3.py (DeepEP on H100)",
         note="superseded by mb2509-h100-dsv3: NeMo 25.09 runs the same model on the same GPUs at 1,297 tok/s/GPU"),
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
    _run("nemo-b200-mixtral-8x7b", "Mixtral-8x7B", "dgx-b200", "fp8", 64, 256, 2, 4096, ep=8, etp=1,
         layout="pinned: " + NEMO_R240 + "recommended_model_configs/model_configs_b200.csv (etp_size 1 or unset at TP 1)",
         metric="tokens_per_gpu", value=16384, source=NEMO_2507),
    _run("nemo-b200-mixtral-8x22b", "Mixtral-8x22B", "dgx-b200", "fp8", 256, 64, 1, 65536, tp=2, pp=4, cp=8, vp=14, ep=8, etp=1,
         layout="pinned: " + NEMO_R240 + "recommended_model_configs/model_configs_b200.csv (etp_size 1 or unset at TP 1)",
         metric="tokens_per_gpu", value=2548, source=NEMO_2507),
    _run("nemo-b200-deepseek-v3", "DeepSeek-V3", "dgx-b200", "fp8", 256, 2048, 1, 4096, tp=2, pp=16, ep=8,
         etp=1, dispatcher="alltoall", metric="tokens_per_gpu", value=1640, source=NEMO_2507, superseded=True,
         layout="pinned: " + NEMO_R240 + "recommended_model_configs/model_configs_b200.csv (etp_size 1) and "
                "llm/pretrain_deepseek_v3.py (alltoall dispatcher with token drop on Blackwell)",
         note="superseded by mb2604-b200-dsv3: 25.09 2,265 and 26.04.01 3,328 tok/s/GPU on the same GPUs"),
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
         dispatcher="deepep", a2a_overlap=True, recompute=["layernorm", "mla_up_proj"],
         layout="pinned: paper Sec. 3.2 (DualPipe, 20 of 132 SMs for communication, node-limited routing) and "
                "Sec. 3.2.3 (recompute of RMSNorm and MLA up-projection)",
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
         metric="mfu", value=0.337, source=REPO + "pretrain-qwen3-30b-moe-bf16/p5en-8nodes", fit=True,
         layout="unverified: expert TP and dispatcher not recorded in config.yaml"),
    # Megatron-Bridge (NeMo 25.09 and 26.04.01 containers), DGX-H100 / B200 / B300; MXFP8 counted as FP8.
    _run("mb2509-h100-dsv3", "DeepSeek-V3", "dgx-h100", "fp8", 1024, 8192, 1, 4096, tp=2, pp=8, vp=4, ep=64, etp=1,
         dedup_by_group=True, dispatcher="hybridep", a2a_overlap=False, recompute=["mla_up_proj", "mlp"],
         metric="tokens_per_gpu", value=1297, source=MB + " (25.09, DeepSeekV3 dropless)", fit=True,
         layout="unverified: the 25.09 recipe is not pinned; the plan equals " + MB_V042 +
                "deepseek/deepseek_workload_base_configs.py DEEPSEEK_V3_PRETRAIN_CONFIG_H100_V1, whose settings are used"),
    _run("mb2604-b200-dsv3", "DeepSeek-V3", "dgx-b200", "fp8", 256, 4096, 1, 4096, pp=8, vp=2, ep=32, etp=1,
         dedup_by_group=True, dispatcher="deepep", a2a_overlap=True, recompute=["mla_up_proj", "mlp"],
         metric="tokens_per_gpu", value=3328, source=MB + " (26.04.01, MXFP8)",
         layout="pinned: " + MB_V042 + "deepseek/deepseek_workload_base_configs.py DEEPSEEK_V3_PRETRAIN_CONFIG_B200_FP8_MX_V2"),
    _run("mb2604-b300-dsv3", "DeepSeek-V3", "dgx-b300", "fp8", 256, 4096, 2, 4096, pp=8, ep=8, etp=1,
         dispatcher="hybridep", a2a_overlap=False, recompute=["mla_up_proj"],
         metric="tokens_per_gpu", value=3456, source=MB + " (26.04.01, MXFP8)", fit=True,
         layout="pinned: " + MB_V042 + "deepseek/deepseek_workload_base_configs.py DEEPSEEK_V3_PRETRAIN_CONFIG_B300_FP8_MX_V2"),
    _run("mb2604-b200-qwen3-30b", "Qwen3-30B-A3B", "dgx-b200", "fp8", 8, 512, 4, 4096, ep=8, etp=1,
         dispatcher="hybridep", a2a_overlap=False,
         metric="tokens_per_gpu", value=27136, source=MB + " (26.04.01, MXFP8)", fit=True,
         layout="pinned: " + MB_V042 + "qwen/qwen3_workload_base_configs.py QWEN3_30B_A3B_PRETRAIN_CONFIG_B200_FP8_MX_V1"),
    _run("mb2604-b300-qwen3-30b", "Qwen3-30B-A3B", "dgx-b300", "fp8", 8, 512, 8, 4096, ep=8, etp=1,
         dispatcher="hybridep", a2a_overlap=False,
         metric="tokens_per_gpu", value=30720, source=MB + " (26.04.01, MXFP8)",
         layout="pinned: " + MB_V042 + "qwen/qwen3_workload_base_configs.py QWEN3_30B_A3B_PRETRAIN_CONFIG_B300_FP8_MX_V1"),
    _run("mb2604-h100-qwen3-30b", "Qwen3-30B-A3B", "dgx-h100", "fp8", 16, 1024, 1, 4096, ep=16, etp=1,
         dispatcher="hybridep", a2a_overlap=False,
         metric="tokens_per_gpu", value=8960, source=MB + " (26.04.01, FP8)",
         layout="pinned: " + MB_V042 + "qwen/qwen3_workload_base_configs.py QWEN3_30B_A3B_PRETRAIN_CONFIG_H100_FP8_CS_V1"),
    _run("mb2604-b200-qwen3-235b", "Qwen3-235B-A22B", "dgx-b200", "fp8", 256, 8192, 1, 4096, pp=8, ep=8, etp=1,
         dispatcher="alltoall", a2a_overlap=False,
         metric="tokens_per_gpu", value=3776, source=MB + " (26.04.01, MXFP8)",
         layout="pinned: " + MB_V042 + "qwen/qwen3_workload_base_configs.py QWEN3_235B_A22B_PRETRAIN_CONFIG_B200_FP8_MX_V2 "
                "(no flex backend: Megatron alltoall dispatcher)"),
    _run("mb2604-b300-qwen3-235b", "Qwen3-235B-A22B", "dgx-b300", "fp8", 256, 8192, 2, 4096, pp=8, ep=8, etp=1,
         dispatcher="hybridep", a2a_overlap=False,
         metric="tokens_per_gpu", value=5936, source=MB + " (26.04.01, MXFP8)",
         layout="pinned: " + MB_V042 + "qwen/qwen3_workload_base_configs.py QWEN3_235B_A22B_PRETRAIN_CONFIG_B300_FP8_MX_V2"),
    _run("mb2604-h100-qwen3-235b", "Qwen3-235B-A22B", "dgx-h100", "fp8", 256, 8192, 1, 4096, tp=2, pp=8, vp=4, ep=32,
         etp=1, dispatcher="alltoall", a2a_overlap=True,
         metric="tokens_per_gpu", value=1712, source=MB + " (26.04.01, FP8)", fit=True,
         layout="pinned: " + MB_V042 + "qwen/qwen3_workload_base_configs.py QWEN3_235B_A22B_PRETRAIN_CONFIG_H100_FP8_CS_V2"),
    # Our own MoE pre-training runs on p6-b300 (DeepEP over EFA, all-to-all overlap on)
    _run("repo-kimi-k2-b300-32n", "Kimi-K2", "p6-b300.48xlarge", "bf16", 256, 256, 4, 4096, tp=8, pp=8, ep=32, etp=1,
         dispatcher="deepep", a2a_overlap=True,
         metric="tokens_per_gpu", value=round(280200 / 256, 1),
         source=REPO + "pretrain-kimi-k2-moe-bf16/p6-b300-32nodes (nvshmem-mb4-overlap-on, 280,200 tok/s)",
         layout="unverified: expert TP 1 assumed (not in config.yaml)"),
    _run("repo-dsv3-b300-32n", "DeepSeek-V3", "p6-b300.48xlarge", "bf16", 256, 256, 4, 4096, tp=8, pp=8, ep=32, etp=1,
         dedup_by_group=True, dispatcher="deepep", a2a_overlap=True,
         metric="tokens_per_gpu", value=round(256 * 4096 / 3.84 / 256, 1),
         source=REPO + "pretrain-deepseek-v3-moe-bf16/p6-b300-32nodes (uccl-mb4-overlap-on, 3.84 s/step)",
         layout="unverified: expert TP 1 assumed (not in config.yaml)"),
]


# Rack-scale hold-out (hardware step 3): NeMo 25.07, DGX-GB200 (NVL72), FP8, the first plan of each table row
# (the parenthesised numbers are other settings). Nothing here is fitted: BASE_EFFICIENCY[("blackwell", "fp8")] and the MoE
# constants come from the RUNS above, none of them from these rows, so each is a pure check of the NVLink-domain model.
# Kept apart from RUNS so the 38-run summary, the fit and every existing number stay as they were.
# Not included: Nemotron5-H-56B (a hybrid Mamba-Transformer: model_shape has no state-space layers).
NEMO_GB200_CSV = ("https://github.com/NVIDIA/NeMo/blob/r2.4.0/scripts/performance/recommended_model_configs/"
                  "model_configs_gb200.csv")
RACK_RUNS = [
    _run("nemo-gb200-llama3-8b", "Llama-3-8B", "dgx-gb200", "fp8", 8, 128, 2, 8192,
         metric="tokens_per_gpu", value=31737, source=NEMO_2507),
    _run("nemo-gb200-llama3-70b", "Llama-3-70B", "dgx-gb200", "fp8", 64, 128, 1, 8192, fsdp=True,
         metric="tokens_per_gpu", value=4066, source=NEMO_2507),
    _run("nemo-gb200-llama31-405b", "Llama-3.1-405B", "dgx-gb200", "fp8", 128, 64, 1, 8192, tp=2, fsdp=True,
         metric="tokens_per_gpu", value=706, source=NEMO_2507),
    _run("nemo-gb200-llama4-scout", "Llama-4-Scout", "dgx-gb200", "fp8", 64, 1024, 1, 8192, ep=16, etp=1,
         metric="tokens_per_gpu", value=13783, source=NEMO_2507,
         layout="unverified: expert TP 1 is in " + NEMO_GB200_CSV + " (llama4 e16, etp_size 1); the token dispatcher is not recorded"),
    _run("nemo-gb200-llama4-maverick", "Llama-4-Maverick", "dgx-gb200", "fp8", 128, 1024, 1, 8192, pp=2, vp=12, ep=64, etp=1,
         metric="tokens_per_gpu", value=11682, source=NEMO_2507,
         layout="unverified: expert TP 1 is in " + NEMO_GB200_CSV + " (llama4 e128, etp_size 1); the token dispatcher is not recorded"),
    _run("nemo-gb200-mixtral-8x7b", "Mixtral-8x7B", "dgx-gb200", "fp8", 64, 256, 2, 4096, ep=8, etp=1,
         layout="pinned: " + NEMO_GB200_CSV + " (etp_size unset at TP 1)",
         metric="tokens_per_gpu", value=17246, source=NEMO_2507),
    _run("nemo-gb200-mixtral-8x22b", "Mixtral-8x22B", "dgx-gb200", "fp8", 256, 64, 1, 65536, tp=2, pp=4, cp=8, vp=14, ep=8, etp=1,
         layout="unverified: no Mixtral-8x22B row in " + NEMO_GB200_CSV + "; expert TP 1 as the DGX-B200 recipe",
         metric="tokens_per_gpu", value=2869, source=NEMO_2507),
    _run("nemo-gb200-deepseek-v3", "DeepSeek-V3", "dgx-gb200", "fp8", 256, 2048, 1, 4096, tp=2, pp=4, ep=64, etp=1,
         dispatcher="alltoall",
         layout="unverified: TP 2, PP 4, EP 64 and expert TP 1 are in " + NEMO_GB200_CSV + "; the alltoall dispatcher is "
                "the one the DGX-B200 25.07 row uses (llm/pretrain_deepseek_v3.py), not checked for GB200",
         metric="tokens_per_gpu", value=2327, source=NEMO_2507),
]


# GB300 against GB200, same recipe (hardware step 5). NeMo / Megatron-Bridge publish the same model, global batch and sequence on
# DGX-GB300 and DGX-GB200 (MXFP8 and FP8 counted as FP8 as for the Megatron-Bridge rows above). The model is checked on the RATIO
# GB300 / GB200 per pair, each system with its own published plan (GB300's larger memory allows other micro batches and plans).
# One constant is fitted on part of the pairs (mfu_model.TGP_EXPONENT, see fit_tgp_exponent): the two dense FP8 pairs of 26.04.01.
# Everything else is hold-out: the dense MXFP8 pairs, the MoE pairs of 26.04.01 and the four MoE pairs of 26.08.01.
# Pairs left out: DeepSeek-V4 Flash and Nemotron 3.5 Lightning / Super / Ultra (architectures not verifiable here: hybrid Mamba or
# unpublished), Kimi-K2 (no GB200 row), and the NVFP4 pairs (no FP4 mode).
NEMO_2608 = "https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-summary.html"
_UNVERIFIED_2608 = "unverified: the 26.08.01 recipe is not pinned (expert TP 1 and the dispatcher are assumed)"
_UNVERIFIED_2604 = "unverified: the 26.04.01 recipe is not pinned (expert TP 1 and the dispatcher are assumed)"
_DENSE = "dense model: no expert or dispatcher settings"


def _pair(name, model, n, gbs, seq, gb300, gb200, *, tag="nemo2608", source=NEMO_2608 + " (26.08.01, MXFP8)",
          layout=_UNVERIFIED_2608, moe=True, prec="fp8", fit=False, fsdp=False):
    """gb300, gb200: dicts of plan fields and the published tokens/s/GPU of each system."""
    rows = []
    for hw, d in (("dgx-gb300", gb300), ("dgx-gb200", gb200)):
        plan = {k: v for k, v in d.items() if k != "value"}
        kw = dict(etp=1) if moe else {}
        if fsdp and plan.get("tp", 1) == 1 and plan.get("pp", 1) == 1:
            kw["fsdp"] = True
        rows.append(_run(f"{tag}-{hw[4:]}-{name}", model, hw, prec, n, gbs, plan.pop("mbs"), seq,
                         metric="tokens_per_gpu", value=d["value"], source=source, layout=layout if moe else "",
                         **kw, **plan))
    return {"id": f"gb300-vs-gb200-{name}" + ("" if tag == "nemo2608" else "-" + tag[4:]), "model": model, "gpus": n,
            "gb300": rows[0], "gb200": rows[1], "fit": fit, "release": tag[4:]}


_MB2604 = MB + " (26.04.01, "

# Fit: the two dense FP8 pairs of 26.04.01 (the same plan on both systems for 70B; pipeline 8 against 16 for 405B).
GB300_FIT_PAIRS = [
    _pair("llama3-70b-fp8", "Llama-3-70B", 64, 256, 8192,
          dict(mbs=2, value=5248), dict(mbs=2, value=4224), tag="nemo2604", source=_MB2604 + "FP8)", moe=False, fit=True, fsdp=True),
    _pair("llama31-405b-fp8", "Llama-3.1-405B", 256, 1536, 8192,
          dict(mbs=1, tp=4, pp=8, vp=4, value=1024), dict(mbs=1, tp=4, pp=16, vp=4, value=864),
          tag="nemo2604", source=_MB2604 + "FP8)", moe=False, fit=True),
]

# Hold-out: the dense MXFP8 pairs and the MoE pairs of 26.04.01, and the four MoE pairs of 26.08.01.
GB300_PAIRS_2604 = [
    _pair("llama3-70b-mxfp8", "Llama-3-70B", 64, 256, 8192,
          dict(mbs=1, pp=4, vp=5, value=4864), dict(mbs=1, tp=2, pp=4, vp=5, value=3712),
          tag="nemo2604", source=_MB2604 + "MXFP8)", moe=False),
    _pair("llama31-405b-mxfp8", "Llama-3.1-405B", 256, 1536, 8192,
          dict(mbs=1, tp=2, pp=8, cp=2, vp=4, value=960), dict(mbs=1, tp=4, pp=16, vp=8, value=800),
          tag="nemo2604", source=_MB2604 + "MXFP8)", moe=False),
    _pair("dsv3", "DeepSeek-V3", 256, 4096, 4096,
          dict(mbs=2, pp=2, vp=8, ep=32, value=4992), dict(mbs=1, pp=4, vp=4, ep=64, value=4256),
          tag="nemo2604", source=_MB2604 + "MXFP8)", layout=_UNVERIFIED_2604),
    _pair("gpt-oss-120b", "GPT-OSS-120B", 64, 1280, 4096,
          dict(mbs=4, ep=64, value=19200), dict(mbs=4, ep=64, value=16640),
          tag="nemo2604", source=_MB2604 + "BF16)", layout=_UNVERIFIED_2604, prec="bf16"),
    _pair("qwen3-30b", "Qwen3-30B-A3B", 8, 512, 4096,
          dict(mbs=8, ep=8, value=31744), dict(mbs=4, ep=8, value=26112),
          tag="nemo2604", source=_MB2604 + "MXFP8)", layout=_UNVERIFIED_2604),
    _pair("qwen3-235b", "Qwen3-235B-A22B", 256, 8192, 4096,
          dict(mbs=2, pp=4, vp=12, ep=32, value=6944), dict(mbs=1, pp=8, vp=3, ep=32, value=5680),
          tag="nemo2604", source=_MB2604 + "MXFP8)", layout=_UNVERIFIED_2604),
]

GB300_PAIRS = [
    _pair("dsv3", "DeepSeek-V3", 256, 4096, 4096,
          dict(mbs=1, pp=2, vp=8, ep=32, value=6288), dict(mbs=1, pp=4, vp=4, ep=64, value=4912)),
    _pair("gpt-oss-120b", "GPT-OSS-120B", 64, 1280, 4096,
          dict(mbs=4, ep=16, value=33024), dict(mbs=4, ep=64, value=28544)),
    _pair("qwen3-30b", "Qwen3-30B-A3B", 8, 512, 4096,
          dict(mbs=8, ep=8, value=44544), dict(mbs=4, ep=8, value=40960)),
    _pair("qwen3-235b", "Qwen3-235B-A22B", 256, 8192, 4096,
          dict(mbs=2, pp=4, vp=12, ep=32, value=8832), dict(mbs=1, pp=8, vp=3, ep=32, value=7264)),
]
ALL_GB300_PAIRS = GB300_FIT_PAIRS + GB300_PAIRS_2604 + GB300_PAIRS

# HGX B300 against HGX B200, same plan on both (NeMo 26.04 / 26.04.01): checks whether the power term (1,100 W against 1,000 W)
# would be right for the B300 as well. Not used for any constant; the term is not applied to the B300 (see mfu_model).
def _b300_pair(name, model, n, gbs, seq, b300, b200, prec="fp8", source=None):
    rows = []
    for hw, d in (("dgx-b300", b300), ("dgx-b200", b200)):
        plan = {k: v for k, v in d.items() if k != "value"}
        rows.append(_run(f"b300pair-{hw[4:]}-{name}", model, hw, prec, n, gbs, plan.pop("mbs"), seq, etp=1,
                         metric="tokens_per_gpu", value=d["value"], source=source, layout=_UNVERIFIED_2604, **plan))
    return {"id": f"b300-vs-b200-{name}", "model": model, "gpus": n, "gb300": rows[0], "gb200": rows[1], "fit": False,
            "release": "2604"}


B300_PAIRS = [
    _b300_pair("dsv3", "DeepSeek-V3", 256, 4096, 4096, dict(mbs=1, pp=16, ep=8, value=3131), dict(mbs=1, pp=16, ep=8, value=2815),
               source=MB + " (26.04, MXFP8)"),
    _b300_pair("gpt-oss-120b", "GPT-OSS-120B", 64, 1280, 4096, dict(mbs=4, ep=8, value=15232), dict(mbs=4, ep=8, value=13568),
               prec="bf16", source=MB + " (26.04.01, BF16)"),
    _b300_pair("qwen3-30b", "Qwen3-30B-A3B", 8, 512, 4096, dict(mbs=8, ep=8, value=30720), dict(mbs=4, ep=8, value=27136),
               source=MB + " (26.04.01, MXFP8)"),
]


def evaluate_b300_pairs(apply_power_term=False):
    """B300 / B200 ratio, published against predicted. apply_power_term=True gives the B300 the board-power term (1,100 W against
    the B200's 1,000 W) for this call only, to see what it would do; the shipped model does not apply it."""
    hw = mm.HARDWARE["dgx-b300"]
    saved = {k: hw.get(k) for k in ("tgp_w", "tgp_ref_w")}
    try:
        if apply_power_term:
            hw["tgp_w"], hw["tgp_ref_w"] = 1100.0, 1000.0
        return evaluate_pairs(B300_PAIRS)
    finally:
        for k, v in saved.items():
            if v is None:
                hw.pop(k, None)
            else:
                hw[k] = v


# Physical bounds of the GB300 / GB200 gap, from NVIDIA's Blackwell Ultra table (Table 2 of
# https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/):
#   dense FP8 5 PFLOPS on both, HBM 8 TB/s on both, board power up to 1,200 W against 1,400 W,
#   SFU exponentials 5 against 10.7 tera-exponentials per second (attention softmax).
SFU_TERA_EXP = {"blackwell": 5.0, "blackwell-ultra": 10.7}


def attention_exp_share(shape, seq, tokens_per_s_per_gpu, tera_exp):
    """Share of one GPU-second the attention exponentials would take at the SFU peak (an upper bound on what a faster SFU can
    save). Per token and layer: one exponential per head and key position, half of them with a causal mask, run in the forward
    pass and again in the backward pass (the softmax is recomputed): heads x seq per layer."""
    a = shape["attention"]
    exps_per_token = a["heads"] * seq * a["full_layers"]
    return tokens_per_s_per_gpu * exps_per_token / (tera_exp * 1e12)


# AWS Trainium 1 (hardware step 6): the only published Trainium training throughput (Neuron 2.24.0, NxD Training, BF16 autocast with an
# FP32 optimizer, micro batch 1, global batch 1024, sequence 8192, 32 x trn1.32xlarge = 1024 NeuronCores = ranks), a table of the Neuron docs.
# Two rows, one source, one model family: they fit ONE constant (BASE_EFFICIENCY[("trainium1", "bf16")], fit_trainium1) and leave nothing for
# a hold-out. Trn2 and Trn3 have no published training numbers at all (docs/CALCULATOR-HARDWARE.md section 7).
# Kept apart from RUNS: the 38-run summary and its fit are unchanged.
NEURON_TRN1 = ("https://awsdocs-neuron.readthedocs-hosted.com/en/v2.27.1/about-neuron/benchmarks/trn1/trn1-training-performance.html "
               "(Neuron 2.24.0, NxDT, BF16 + FP32 optimizer)")
TRN1_RUNS = [
    _run("neuron-trn1-llama31-8b", "Llama-3-8B", "trn1.32xlarge", "bf16", 1024, 1024, 1, 8192, tp=32,
         metric="tokens_per_gpu", value=round(47.95 * 8192 / 1024, 2), source=NEURON_TRN1, fit=True,
         note="TP 32 DP 32 PP 1 ZeRO-1; 47.95 seq/s; Llama 3.1 8B has the shape of Llama 3 8B"),
    _run("neuron-trn1-llama31-70b", "Llama-3-70B", "trn1.32xlarge", "bf16", 1024, 1024, 1, 8192, tp=32, pp=8,
         metric="tokens_per_gpu", value=round(7.94 * 8192 / 1024, 2), source=NEURON_TRN1, fit=True,
         note="TP 32 DP 4 PP 8; 7.94 seq/s; Llama 3.1 70B has the shape of Llama 3 70B"),
]

# Rows added with mfu-model-3 (reported as their own group).
NEW_MOE_RUNS = {"mb2509-h100-dsv3", "mb2604-b200-dsv3", "mb2604-b300-dsv3", "mb2604-b200-qwen3-30b",
                "mb2604-b300-qwen3-30b", "mb2604-h100-qwen3-30b", "mb2604-b200-qwen3-235b", "mb2604-b300-qwen3-235b",
                "mb2604-h100-qwen3-235b", "repo-kimi-k2-b300-32n", "repo-dsv3-b300-32n"}
# Grid for the default EP all-to-all overlap (plans without overlap_ep), fitted with the MoE constants.
FIT_OVERLAP_EP = (0.0, 0.3, 0.5, 0.7, 0.9)


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
                     "fit": r["fit"], "source": r["source"], "note": r["note"],
                     "superseded": r.get("superseded", False), "new_moe": r["id"] in NEW_MOE_RUNS,
                     "layout_unverified": str(r.get("layout", "")).startswith("unverified")})
    return rows


def summary(rows):
    """Scored rows only (superseded rows are printed, not scored). `original` = the first 29-run set
    without its superseded rows; `new_moe` = the MoE rows added with mfu-model-3."""
    def stats(sel):
        errs = [abs(x["error_pct"]) for x in sel]
        return {"runs": len(errs), "median_abs_error_pct": round(statistics.median(errs), 1) if errs else None,
                "max_abs_error_pct": round(max(errs), 1) if errs else None}
    scored = [x for x in rows if not x.get("superseded")]
    return {"all": stats(scored), "hold_out": stats([x for x in scored if not x["fit"]]),
            "fit": stats([x for x in scored if x["fit"]]),
            "original": stats([x for x in scored if not x.get("new_moe")]),
            "original_hold_out": stats([x for x in scored if not x.get("new_moe") and not x["fit"]]),
            "new_moe": stats([x for x in scored if x.get("new_moe")]),
            "new_moe_hold_out": stats([x for x in scored if x.get("new_moe") and not x["fit"]]),
            "superseded": stats([x for x in rows if x.get("superseded")])}


def evaluate_pairs(pairs=GB300_PAIRS, tgp_exponent=None):
    """Per pair: published and predicted GB300 / GB200 throughput ratio and the error of the ratio.
    tgp_exponent: evaluate with another power exponent (0 = the model before the power term)."""
    out = []
    saved = mm.TGP_EXPONENT
    if tgp_exponent is not None:
        mm.TGP_EXPONENT = tgp_exponent
    try:
        return _evaluate_pairs(pairs, out)
    finally:
        mm.TGP_EXPONENT = saved


def _evaluate_pairs(pairs, out):
    for p in pairs:
        a, b = evaluate([p["gb300"]])[0], evaluate([p["gb200"]])[0]
        published, predicted = a["reported"] / b["reported"], a["predicted"] / b["predicted"]
        out.append({"id": p["id"], "model": p["model"], "gpus": p["gpus"], "fit": p["fit"], "release": p["release"],
                    "published_gb300": a["reported"], "published_gb200": b["reported"],
                    "predicted_gb300": a["predicted"], "predicted_gb200": b["predicted"],
                    "published_ratio": round(published, 3), "predicted_ratio": round(predicted, 3),
                    "ratio_error_pct": round((predicted / published - 1) * 100, 1),
                    "gb300_error_pct": a["error_pct"], "gb200_error_pct": b["error_pct"]})
    return out


def pair_summary(rows):
    errs = [abs(x["ratio_error_pct"]) for x in rows]
    return {"pairs": len(errs), "median_abs_error_pct": round(statistics.median(errs), 1) if errs else None,
            "max_abs_error_pct": round(max(errs), 1) if errs else None}


def pair_groups(rows):
    """Summaries of the fit pairs and of the hold-out pairs (all, dense, MoE of 26.04.01, MoE of 26.08.01)."""
    hold = [x for x in rows if not x["fit"]]
    moe = lambda x: x["model"] in ("DeepSeek-V3", "GPT-OSS-120B", "Qwen3-30B-A3B", "Qwen3-235B-A22B")  # noqa: E731
    return {"fit": pair_summary([x for x in rows if x["fit"]]), "hold_out": pair_summary(hold),
            "hold_out_dense": pair_summary([x for x in hold if not moe(x)]),
            "hold_out_moe_2604": pair_summary([x for x in hold if moe(x) and x["release"] == "2604"]),
            "hold_out_moe_2608": pair_summary([x for x in hold if moe(x) and x["release"] == "2608"])}


def fit_tgp_exponent(pairs=GB300_FIT_PAIRS):
    """TGP_EXPONENT: least squares of the log ratio error on the fit pairs. The ratio is predicted with the exponent 0 (no
    power term, everything else of the model) and the exponent is what turns that into the published ratio:
        ln(published / predicted_0) = exponent x ln(tgp_w / tgp_ref_w)"""
    saved = mm.TGP_EXPONENT
    mm.TGP_EXPONENT = 0.0
    try:
        gaps = []
        for p in pairs:
            a, b = predict(p["gb300"])[0], predict(p["gb200"])[0]
            gaps.append(math.log((p["gb300"]["reported"] / p["gb200"]["reported"]) / (a / b)))
    finally:
        mm.TGP_EXPONENT = saved
    hw = mm.HARDWARE["dgx-gb300"]
    return round(sum(gaps) / (len(gaps) * math.log(hw["tgp_w"] / hw["tgp_ref_w"])), 2)


def pairs_markdown(rows):
    out = ["| Pair | GPUs | GB300 reported / predicted | GB200 reported / predicted | Ratio reported | Ratio predicted | "
           "Ratio error | GB300 error | GB200 error | Set |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        out.append(f"| {x['id']} | {x['gpus']} | {x['published_gb300']:,.0f} / {x['predicted_gb300']:,.0f} | "
                   f"{x['published_gb200']:,.0f} / {x['predicted_gb200']:,.0f} | {x['published_ratio']:.3f} | "
                   f"{x['predicted_ratio']:.3f} | {x['ratio_error_pct']:+.1f}% | {x['gb300_error_pct']:+.1f}% | "
                   f"{x['gb200_error_pct']:+.1f}% | {'fit' if x['fit'] else 'hold-out'} |")
    return "\n".join(out)


def fit_trainium1():
    """BASE_EFFICIENCY[("trainium1", "bf16")]: geometric mean of the per-row solutions of the two Neuron Trn1 rows."""
    saved = dict(mm.BASE_EFFICIENCY)
    shapes, sols = {}, []
    try:
        for r in TRN1_RUNS:
            def f(x, r=r):
                mm.BASE_EFFICIENCY[("trainium1", "bf16")] = x
                return predict(r, shapes)[0]
            sols.append(_solve(f, r["reported"]))
    finally:
        mm.BASE_EFFICIENCY.clear()
        mm.BASE_EFFICIENCY.update(saved)
    return round(math.exp(statistics.mean(math.log(v) for v in sols)), 3)


def trn1_summary(rows):
    errs = [abs(x["error_pct"]) for x in rows]
    return {"runs": len(errs), "median_abs_error_pct": round(statistics.median(errs), 1), "max_abs_error_pct": round(max(errs), 1)}


def rack_summary(rows):
    """Median and worst absolute error of the rack-scale hold-out rows (evaluate(RACK_RUNS))."""
    errs = [abs(x["error_pct"]) for x in rows]
    return {"runs": len(errs), "median_abs_error_pct": round(statistics.median(errs), 1) if errs else None,
            "max_abs_error_pct": round(max(errs), 1) if errs else None}


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
    """Fit BASE_EFFICIENCY per (family, precision) on dense fit runs, then MOE_EXPERT_FACTOR,
    MOE_EXPERT_HALF_FFN and the default EP overlap (OVERLAP["ep"]) on the MoE fit runs.
    Each cell is the geometric mean of the per-run solutions."""
    saved_base, saved_moe, saved_half = dict(mm.BASE_EFFICIENCY), mm.MOE_EXPERT_FACTOR, mm.MOE_EXPERT_HALF_FFN
    saved_ov = mm.OVERLAP["ep"]
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
    for ov in FIT_OVERLAP_EP:
        mm.OVERLAP["ep"] = ov
        for fmax in [x / 100 for x in range(30, 101, 1)]:
            for half in (0, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096):
                mm.MOE_EXPERT_FACTOR, mm.MOE_EXPERT_HALF_FFN = fmax, half
                loss = sum(math.log(predict(r, shapes)[0] / r["reported"]) ** 2 for r in moe_runs)
                if best is None or loss < best[0]:
                    best = (loss, fmax, half, ov)
    moe_factor, moe_half, ov_ep = (best[1], best[2], best[3]) if best else (saved_moe, saved_half, saved_ov)
    mm.BASE_EFFICIENCY.clear()
    mm.BASE_EFFICIENCY.update(saved_base)
    mm.MOE_EXPERT_FACTOR, mm.MOE_EXPERT_HALF_FFN = saved_moe, saved_half
    mm.OVERLAP["ep"] = saved_ov
    return {"base_efficiency": {f"{k[0]}/{k[1]}": v for k, v in sorted(fitted.items())},
            "moe_expert_factor": moe_factor, "moe_expert_half_ffn": moe_half, "overlap_ep": ov_ep,
            "tgp_exponent": fit_tgp_exponent(),
            "trainium1_bf16": fit_trainium1(),
            "fit_runs": [r["id"] for r in RUNS if r["fit"]]}


def days_per_trillion(row, value):
    """Days to train 1T tokens on the run's GPUs at a tokens/s/GPU value (None for MFU-only runs)."""
    if row["metric"] == "mfu":
        return None
    return 1e12 / (value * row["gpus"]) / 86400


def _set(x):
    if x.get("superseded"):
        return "superseded"
    s = "fit" if x["fit"] else "hold-out"
    return s + (", layout unverified" if x.get("layout_unverified") else "")


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
                   f"{x['predicted_mfu'] * 100:.1f}% | {days} | {_set(x)} |")
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
                     f"<td>{x['predicted_mfu'] * 100:.1f} pct</td><td>{days}</td><td>{_set(x)}</td></tr>")
    return "<table>" + "".join(cells) + "</table>"


if __name__ == "__main__":
    if "--fit" in sys.argv:
        print(json.dumps(fit(), indent=1))
        sys.exit(0)
    rows = evaluate()
    print(markdown(rows))
    print(json.dumps(summary(rows), indent=1))
    rack_rows = evaluate(RACK_RUNS)
    print("\nRack-scale hold-out (DGX GB200, NeMo 25.07)\n")
    print(markdown(rack_rows))
    print(json.dumps(rack_summary(rack_rows), indent=1))
    trn_rows = evaluate(TRN1_RUNS)
    print("\nAWS Trainium 1, the two Neuron rows (fit, no hold-out)\n")
    print(markdown(trn_rows))
    print(json.dumps(trn1_summary(trn_rows), indent=1))
    pair_rows = evaluate_pairs(ALL_GB300_PAIRS)
    print("\nGB300 vs GB200, same recipe (NeMo 26.04.01 and 26.08.01)\n")
    print(pairs_markdown(pair_rows))
    print(json.dumps(pair_groups(pair_rows), indent=1))
