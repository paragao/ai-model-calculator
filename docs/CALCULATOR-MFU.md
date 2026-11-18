# Training calculator: expected MFU, bubble, throughput and cost

Code: `utils/mfu_model.py` (one module, stdlib only, no globals from elsewhere), validation set
`utils/validation_runs.py`, tests `tests/test_mfu_model.py`. `utils/phase9_mfu_estimate.py` uses it for
the plan search, MFU, throughput, days and cost (`python3 ai_model_calculator.py --estimate`). The
memory phases (`utils/phase1_memory.py`, `utils/core_calculations.py`) still give the memory per GPU.

This page lists every formula and constant of the model. The same files are used, line for line, by the
portal calculator Lambda. Model version: `mfu-model-5` (B300 board-power term and joint refit; mfu-model-4 numbers are in
`docs/CALCULATOR-HARDWARE.md` section 6a).

## 1. What changed from the old calculator

| Before | Now |
|---|---|
| MFU fixed at 40% for every model, GPU and plan | MFU estimated from the architecture, plan and hardware (sections 3-5) |
| tokens/s = 0.40 x BF16 peak x GPUs / (6 x active params): time x GPUs constant, so the same cost at any GPU count | step time includes bubble and exposed communication, so bigger plans can cost more |
| Bubble from a global batch of micro_batch x DP (0 or 1 micro-batches, so 100% bubbles) | global batch of 16M tokens (Llama 3 405B) unless given; bubble (PP-1)/(VP x m + PP-1) |
| Plan = smallest TP x PP that fits; EP never searched | every TP / PP / EP plan that divides the GPUs, scored by estimated throughput; the fastest that fits wins |
| B300 BF16 peak 3,375 TFLOPS | 2,250 (p6-b300.48xlarge: the dense FP8 and BF16 of p6-b200.48xlarge, 1.5x only at FP4, [AWS P6 page](https://aws.amazon.com/ec2/instance-types/p6/)) |
| Expert optimizer state sharded over the full DP when EP = DP | sharded over DP / EP (Megatron expert-data-parallel group) |
| FP8 used the BF16 peak | FP8 uses the FP8 dense peak and its own efficiency |

On the first 29 published runs the old formula is off by a median of 38% (max 144%). The model is off
by a median of 5.7% (max 17.8%) on 27 of them (two are superseded, section 7), and by 7.7% (max 38.2%)
on 11 newer MoE runs.

## 2. Conventions

- Plan layout (Megatron-Core): GPUs = TP x PP x CP x DP. EP is carved out of DP:
  GPUs / PP = ETP x EP x expert-DP, with expert TP (ETP) = TP when TP x EP divides GPUs / PP, else 1.
- m = micro-batches per DP rank per step = GBS / (DP x MBS).
- Peak = dense tensor-core peak of the training precision (BF16 or FP8).
- MFU = model FLOPs per step / (GPUs x peak x step time).

## 3. Shape from any config.json (`model_shape`)

One parser for dense and MoE, MHA / GQA / MQA / MLA, linear and hybrid attention, gated or plain
FFN, shared and routed experts, MoE layer placement, tied embeddings. Field names are Hugging Face's.

| Quantity | Formula |
|---|---|
| FFN matrices `k` | 2 if `activation_function` is set without "glu" (GPT-2/OPT style) or `hidden_act` is not gated (gelu, relu, relu2); 3 for silu/swish/swiglu/geglu/gelu_pytorch_tanh or unknown |
| Attention, GQA/MHA/MQA, per layer | d x head_dim x (2 H + 2 KV); head_dim = `head_dim` if > 0 else d / H; KV = 1 if `multi_query` |
| Attention, MLA (`kv_lora_rank` set) | q = d q_lora + q_lora H (nope + rope) (or d H qk without q_lora); kv = d (kv_lora + rope) + kv_lora H (nope + v); o = H v d |
| Linear attention layers | `linear_attn_layers` of the L layers use 4 d x `linear_attn_heads` x `linear_attn_head_dim` (q, k, v, o) |
| Experts | E = `n_routed_experts` / `num_experts` / `num_local_experts`; top-k = `num_experts_per_tok`; expert FFN = `moe_intermediate_size` |
| Shared experts | width = `shared_expert_intermediate_size`, else (`n_shared_experts` or `num_shared_experts`) x expert FFN, else `moe_shared_expert_intermediate_size` |
| MoE layers | `moe_layers`, else L / `decoder_sparse_step` (or `interleave_moe_layer_step`, `moe_layer_freq`), else L - `first_k_dense_replace` |
| Total | attention + dense layers x k d ffn + MoE layers x (E k d f_e + k d f_shared + d E) + vocab d (x2 if untied) |
| Active | same with top-k experts instead of E |
| MTP | `num_nextn_predict_layers` x (one layer + 2 d d + vocab d) added to compute only |

With an exact count (safetensors index) all counts are scaled by count / shape total.
Check: 11 published architectures within 2% (total) and 3% (active) of their published sizes
(`test_every_validation_architecture_matches_its_published_size`).

## 4. FLOPs per token (PaLM appendix B convention)

F = 6 x N_matmul + 6 x H x S x (d_qk + d_v) x L_full + 12 x H_lin x d_lin^2 x L_linear

N_matmul = active parameters minus the input embedding (a tied LM head still counts). Attention is
counted non-causal (PaLM, Megatron). f_e = expert share of F = 6 x active expert params / F.

## 5. Step time and MFU (`estimate`)

```
t_compute = F x tokens_per_step / GPUs / (peak x eta) x (1 + recompute share) / (1 - comm SM share)
eta       = 1 / ((1 - f_e) / eta_base + f_e / (eta_base x moe_factor))
moe_factor= MOE_EXPERT_FACTOR x f_expert / (f_expert + half),  half = MOE_EXPERT_HALF_FFN_FAMILY[family] or MOE_EXPERT_HALF_FFN
eta_base  = BASE_EFFICIENCY[family, precision] x (tgp_w / tgp_ref_w) ^ TGP_EXPONENT   (board power: GB300, B300)
bubble    = (PP - 1) / (VP x m + PP - 1)               (Narayanan et al. 2021)
t_pipe    = t_compute / (1 - bubble)
exposed(x)= max(t_x x (1 - OVERLAP_x), t_x - t_compute)
t_step    = t_pipe + exposed(TP) + exposed(CP) + exposed(EP) + exposed(DP)
MFU       = F x tokens_per_step / (GPUs x peak x t_step)
tokens/s  = tokens_per_step / t_step;  wall-clock tokens/s = tokens/s x GOODPUT
days      = tokens / wall-clock tokens/s / 86400;  cost = days x 24 x nodes x price per node-hour
```

Recompute share = forward FLOPs of the plan's `recompute` modules (Megatron-Core `recompute_modules`) / F:
`mlp` 2 x dense-layer MLP params, `moe` 2 x active MoE-block params, `shared_experts`, `mla_up_proj`
2 x (q_lora H (nope + rope) + kv_lora H (nope + v)) per full-attention layer, `core_attn` the forward third
of the attention-score FLOPs; `layernorm` and `moe_act` add no matmul FLOPs. MFU still counts model
FLOPs only, so recompute lowers MFU.

Comm SM share = `dispatcher_sms` (default COMM_SMS = 20) / SMs per GPU, only when the plan's
`dispatcher` is an SM-kernel all-to-all (DeepEP, HybridEP) and `a2a_overlap` is set: those SMs run the
all-to-all while the GEMMs run ([DeepSeek-V3 paper](https://arxiv.org/abs/2412.19437) Sec. 3.2.2: 20 of
132 H800 SMs). The calculator's own plans set neither field, so they are unchanged.

Communication volumes per GPU per step (bytes; BF16 on the wire), time = bytes / bandwidth:

| Part | Bytes | Bandwidth |
|---|---|---|
| TP (sequence parallel) | 8 (TP-1)/TP x MBS x S/CP x d x 2 x L/PP x m | NVLink if TP <= GPUs per node |
| CP (ring attention) | 3 (CP-1) x MBS x S/CP x 2 KV d_qk x 2 x L_full/PP x m | group bandwidth (below) |
| EP all-to-all | 4 x (MBS S/CP/TP) x top-k x d x 2 x (EP-1)/EP x MoE layers/PP x m; off-node share 1 - (node GPUs / ETP)/EP over the network, the rest over NVLink, the slower decides | NVLink and network |
| EP with node de-duplication (plan `dedup_by_group`, DeepEP) | network copies = min(top-k, `topk_group`) when `n_group` > 1 | |
| DP (distributed optimizer) | (g-1)/g x local params x (2 + 2), dense g = DP x CP, expert g = GPUs/PP/(ETP x EP), local = params / (TP x PP) (experts / (ETP x PP x EP)) | group bandwidth |
| ZeRO-2 | grads reduce-scattered every micro-batch: t_DP / 2 x (1 + m) | |
| ZeRO-3 = FSDP (plan `zero` 3 or `fsdp`) | per layer per micro-batch: params gathered before forward and again before backward, grads reduce-scattered after backward, 3 Psi per micro-batch ([ZeRO paper](https://arxiv.org/abs/1910.02054) sec. 7.2.2), so 1.5 x m x t_DP per step; exposed time per layer, see below | |

ZeRO-3 / FSDP exposed time is summed per unit u (dense layer, MoE layer, embedding + LM head), each
micro-batch, because a gather can only hide behind the neighbouring unit's compute in the same pass
(about 1/3 of the unit's training compute c_u forward, 2/3 backward), and only as far as the prefetch
buffer allows:

```
AG_u = (g-1)/g x params_u x 2 / bw(g);  RS_u = the same for grads
exposed_fwd = max(AG_u x (1 - OVERLAP_fsdp), AG_u - k x c_u / 3)
exposed_bwd = max((AG_u + RS_u) x (1 - OVERLAP_fsdp), AG_u + RS_u - k x 2 c_u / 3)
k = (live params - largest unit) / largest unit, clipped to 0..1   (plan field zero3_prefetch)
```

`live params` is the memory model's ZeRO-3 transient: largest unit + prefetch bucket (5e8), capped at
`stage3_max_live_parameters` (1e9) ([DeepSpeed config](https://www.deepspeed.ai/docs/config-json/)).
So memory and time describe the same schedule: a unit larger than the cap (GLM-5.3-Flash at EP 1:
7.4B params per MoE layer) cannot prefetch, and every gather of it is exposed. The validation runs
(Megatron FSDP) use k = 1.

Group bandwidth: inside one node NVLink; across nodes min(NVLink, r x NIC per GPU), r = GPUs of the
group per node (= GPUs per node / stride). Each x LINK_EFFICIENCY. Pipeline send/recv is hidden in
steady state (1F1B); its fill and drain is the bubble term.

## 6. Constants

| Constant | Value | Source / how |
|---|---|---|
| Peak BF16 / FP8 dense TFLOPS | H100, H200, H800 989 / 1979; A100 312; B200, B300 2250 / 4500 | NVIDIA datasheets; B300: equal to B200 except FP4 ([AWS P6 page](https://aws.amazon.com/ec2/instance-types/p6/)) |
| NVLink per direction, GB/s | H100/H200 450, B200/B300 900, A100 300, H800 200 | half of total NVLink bandwidth per GPU |
| Network per GPU, GB/s | p5/p5e/p5en/p6-b200 50 (3200 Gbps EFA / 8), p6-b300 100 (6400 Gbps), p4d/p4de 6.25 (400 Gbps), GB200 100 (3200 Gbps per 4-GPU instance), GB300 100 (8 x 400 Gbps per 4-GPU server, paragao 2026-11-17); DGX H100/B200 50, DGX B300 100 (8 x 800 Gb/s ConnectX-8), DGX A100 25, Meta Llama 3 RoCE 50, DeepSeek H800 IB 50 | AWS instance pages; NVIDIA DGX datasheets; Llama 3 paper Sec. 3.3.1; DeepSeek-V3 paper Sec. 3.1 |
| BASE_EFFICIENCY hopper/bf16, hopper/fp8, blackwell/bf16, blackwell/fp8, ampere/bf16 | 0.573, 0.45, 0.417, 0.417, 0.471 | fitted (geometric mean per cell) on the dense fit runs, with the board-power term in place (blackwell/bf16 was 0.443 in mfu-model-4) |
| MOE_EXPERT_FACTOR, MOE_EXPERT_HALF_FFN | 0.78, 1024 | the mfu-model-4 fit (grid, least squares of log error, all 6 MoE fit runs), kept in mfu-model-5: `fit_moe_shared` run on `mfu_model_4()` |
| MOE_EXPERT_HALF_FFN_FAMILY | blackwell 1664 (others use MOE_EXPERT_HALF_FFN) | fitted (grid of 64, least squares of log error) on the 2 Blackwell MoE fit runs with the shared constants held (`fit_blackwell_moe_half`); `docs/CALCULATOR-HARDWARE.md` 6a |
| LINK_EFFICIENCY nvlink / net | 0.76 / 0.80 | nvlink: our DeepEP intranode 340 GB/s vs 450 (benchmarks/ep-dispatch-combine-bf16/p5en-2nodes); net: assumption |
| OVERLAP tp / cp / ep / dp / fsdp | 0.5 / 0.9 / 0.3 / 0.9 / 0.8 | assumptions, except EP: fitted with the MoE constants (grid 0 / 0.3 / 0.5 / 0.7 / 0.9); a plan may set `overlap_ep` (DualPipe 0.95) |
| SMS_PER_GPU | H100 / H200 / H800 132, A100 108, B200 148, B300 160 | NVIDIA H100 and A100 whitepapers; [B200: 74 per die](https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut); [B300: up to 160](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/) |
| TGP_EXPONENT | 1.25 | fitted on the two dense FP8 GB300 / GB200 pairs of NeMo 26.04.01; applies to systems with `tgp_w` and `tgp_ref_w` (GB300: 1,400 W against 1,200 W; B300: 1,100 W against 1,000 W since mfu-model-5); section 7b |
| tgp_w, tgp_ref_w (B300) | 1100, 1000 | "TDP per GPU" 1,000 W (B200) and 1,100 W (B300): [arXiv 2610.41919](https://arxiv.org/abs/2610.41919); AWS and NVIDIA publish no B300 per-GPU power figure. Hardware: `p6-b200.48xlarge`, `p6-b300.48xlarge` ([AWS instance table](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html)) |
| COMM_SMS | 20 | DeepSeek-V3 paper Sec. 3.2.2 |
| DEFAULT_TOKENS_PER_STEP | 16,777,216 (16M) | Llama 3 405B, Llama 3 paper Table 4 |
| GOODPUT | 0.90 | Llama 3 paper Sec. 3.3.4 (> 90% effective training time) |

No constant is per model. Fit runs (15): NeMo Llama3 8B/70B on H100 and B200, NeMo Mixtral 8x7B on
H100, MT-NLG 530B at 2240 GPUs, our Llama 70B BF16 p5en and p6-b200, Qwen3-8B p5en and p6-b300,
Qwen3-30B-A3B p5en; Megatron-Bridge DeepSeek-V3 on H100 (25.09) and B300, Qwen3-30B-A3B on B200 and
Qwen3-235B-A22B on H100 (26.04.01). Everything else (23 scored runs) is hold-out.
Refit: `python3 validation_runs.py --fit`; the test fails if the constants differ from the fit. The fit order (mfu-model-5) is in
`validation_runs.fit`: dense base cells, then the shared MoE constants (kept), then the Blackwell half width, then the exponent.

## 7. Validation set: predicted vs published

40 runs, 13 model families (Llama 3 8B/70B/405B, Mixtral 8x7B/8x22B, DeepSeek-V3, GPT-3 175B,
Nemotron-4 340B, MT-NLG 530B, Qwen3 8B, 30B-A3B and 235B-A22B, Kimi-K2), H100 / H200 / H800 / B200 /
B300 / A100, BF16 and FP8, 8 to 16,384 GPUs. 38 are scored; 2 are superseded (printed, not fitted or
scored). `python3 validation_runs.py` prints this table.

Every MoE run states where its layout comes from (`layout` in `validation_runs.py`): `pinned` = expert
TP, token dispatcher and recompute read from the recipe of the release that published the number
([NeMo r2.4.0](https://github.com/NVIDIA/NeMo/tree/r2.4.0/scripts/performance) for 25.07,
[Megatron-Bridge v0.4.2](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/v0.4.2/scripts/performance/configs),
which carries the 26.04.01 summary); `unverified` = assumed (marked in the Set column). The NeMo 25.07
recipes set expert TP 1 for DeepSeek-V3 and Mixtral 8x22B; until mfu-model-3 these rows used expert TP =
TP, which spread EP over two nodes and hid most of the DeepSeek-V3 error behind network time.

| Run | Model | GPUs | Hardware | Prec | Metric | Reported | Predicted | Error | Predicted MFU | Days per 1T tokens, reported / predicted | Set |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nemo-h100-llama3-8b | Llama-3-8B | 8 | dgx-h100 | fp8 | tokens_per_gpu | 14,744 tok/s/GPU | 14,697 tok/s/GPU | -0.3% | 43.0% | 98.1 / 98.4 | fit |
| nemo-h100-llama3-70b | Llama-3-70B | 64 | dgx-h100 | fp8 | tokens_per_gpu | 1,663 tok/s/GPU | 1,671 tok/s/GPU | +0.5% | 40.6% | 108.7 / 108.2 | fit |
| nemo-h100-llama31-405b | Llama-3.1-405B | 1024 | dgx-h100 | fp8 | tokens_per_gpu | 320 tok/s/GPU | 297 tok/s/GPU | -7.3% | 39.3% | 35.3 / 38.1 | hold-out |
| nemo-h100-mixtral-8x7b | Mixtral-8x7B | 64 | dgx-h100 | fp8 | tokens_per_gpu | 8,275 tok/s/GPU | 7,801 tok/s/GPU | -5.7% | 32.7% | 21.9 / 23.2 | fit |
| nemo-h100-mixtral-8x22b | Mixtral-8x22B | 256 | dgx-h100 | fp8 | tokens_per_gpu | 1,280 tok/s/GPU | 1,412 tok/s/GPU | +10.3% | 36.0% | 35.3 / 32.0 | hold-out |
| nemo-h100-deepseek-v3 | DeepSeek-V3 | 1024 | dgx-h100 | fp8 | tokens_per_gpu | 887 tok/s/GPU | 1,300 tok/s/GPU | +46.6% | 19.1% | 12.7 / 8.7 | superseded |
| nemo-h100-gpt3-175b | GPT-3-175B | 512 | dgx-h100 | fp8 | tokens_per_gpu | 824 tok/s/GPU | 760 tok/s/GPU | -7.8% | 41.3% | 27.4 / 29.8 | hold-out |
| nemo-h100-nemotron4-340b | Nemotron-4-340B | 256 | dgx-h100 | fp8 | tokens_per_gpu | 335 tok/s/GPU | 360 tok/s/GPU | +7.4% | 38.3% | 135.0 / 125.6 | hold-out |
| nemo-b200-llama3-8b | Llama-3-8B | 8 | dgx-b200 | fp8 | tokens_per_gpu | 30,131 tok/s/GPU | 32,369 tok/s/GPU | +7.4% | 41.7% | 48.0 / 44.7 | fit |
| nemo-b200-llama3-70b | Llama-3-70B | 64 | dgx-b200 | fp8 | tokens_per_gpu | 3,690 tok/s/GPU | 3,463 tok/s/GPU | -6.2% | 37.0% | 49.0 / 52.2 | fit |
| nemo-b200-llama31-405b | Llama-3.1-405B | 128 | dgx-b200 | fp8 | tokens_per_gpu | 674 tok/s/GPU | 657 tok/s/GPU | -2.5% | 38.3% | 134.2 / 137.7 | hold-out |
| nemo-b200-mixtral-8x7b | Mixtral-8x7B | 64 | dgx-b200 | fp8 | tokens_per_gpu | 16,384 tok/s/GPU | 16,154 tok/s/GPU | -1.4% | 29.8% | 11.0 / 11.2 | hold-out |
| nemo-b200-mixtral-8x22b | Mixtral-8x22B | 256 | dgx-b200 | fp8 | tokens_per_gpu | 2,548 tok/s/GPU | 3,001 tok/s/GPU | +17.8% | 33.6% | 17.7 / 15.1 | hold-out |
| nemo-b200-deepseek-v3 | DeepSeek-V3 | 256 | dgx-b200 | fp8 | tokens_per_gpu | 1,640 tok/s/GPU | 3,494 tok/s/GPU | +113.0% | 22.6% | 27.6 / 12.9 | superseded |
| nemo-b200-gpt3-175b | GPT-3-175B | 512 | dgx-b200 | fp8 | tokens_per_gpu | 1,523 tok/s/GPU | 1,586 tok/s/GPU | +4.2% | 37.9% | 14.8 / 14.3 | hold-out |
| nemo-b200-nemotron4-340b | Nemotron-4-340B | 128 | dgx-b200 | fp8 | tokens_per_gpu | 632 tok/s/GPU | 725 tok/s/GPU | +14.7% | 33.9% | 143.1 / 124.8 | hold-out |
| meta-llama3-405b-8k-gpus | Llama-3.1-405B | 8192 | meta-h100-roce | bf16 | mfu | 43.0% | 50.5% | +17.4% | 50.5% | – | hold-out |
| meta-llama3-405b-16k-gpus | Llama-3.1-405B | 16384 | meta-h100-roce | bf16 | mfu | 41.0% | 47.9% | +16.8% | 47.9% | – | hold-out |
| deepseek-v3-paper | DeepSeek-V3 | 2048 | deepseek-h800 | fp8 | tokens_per_gpu_wallclock | 1,543 tok/s/GPU | 1,504 tok/s/GPU | -2.5% | 24.6% | 3.7 / 3.8 | hold-out |
| mtnlg-530b-2240-gpus | MT-NLG-530B | 2240 | dgx-a100-40g | bf16 | mfu | 40.4% | 40.4% | -0.0% | 40.4% | – | fit |
| mtnlg-530b-2800-gpus | MT-NLG-530B | 2800 | dgx-a100-40g | bf16 | mfu | 38.8% | 39.2% | +1.0% | 39.2% | – | hold-out |
| mtnlg-530b-3360-gpus | MT-NLG-530B | 3360 | dgx-a100-40g | bf16 | mfu | 36.2% | 38.0% | +5.0% | 38.0% | – | hold-out |
| repo-llama70b-bf16-p5en | Llama-3-70B | 64 | p5en.48xlarge | bf16 | tokens_per_gpu | 1,200 tok/s/GPU | 1,133 tok/s/GPU | -5.6% | 55.2% | 150.7 / 159.6 | fit |
| repo-llama70b-fp8-p5en | Llama-3-70B | 64 | p5en.48xlarge | fp8 | tokens_per_gpu | 1,831 tok/s/GPU | 1,679 tok/s/GPU | -8.3% | 40.8% | 98.8 / 107.7 | hold-out |
| repo-llama70b-bf16-b200 | Llama-3-70B | 64 | p6-b200.48xlarge | bf16 | tokens_per_gpu | 1,860 tok/s/GPU | 1,847 tok/s/GPU | -0.7% | 39.5% | 97.2 / 97.9 | fit |
| repo-llama70b-fp8-b200 | Llama-3-70B | 64 | p6-b200.48xlarge | fp8 | tokens_per_gpu | 3,474 tok/s/GPU | 3,463 tok/s/GPU | -0.3% | 37.0% | 52.1 / 52.2 | hold-out |
| repo-qwen3-8b-bf16-p5en | Qwen3-8B | 16 | p5en.48xlarge | bf16 | tokens_per_gpu | 10,125 tok/s/GPU | 10,728 tok/s/GPU | +6.0% | 57.1% | 71.4 / 67.4 | fit |
| repo-qwen3-8b-bf16-b300 | Qwen3-8B | 16 | p6-b300.48xlarge | bf16 | tokens_per_gpu | 19,875 tok/s/GPU | 20,014 tok/s/GPU | +0.7% | 46.8% | 36.4 / 36.1 | fit |
| repo-qwen3-30b-a3b-bf16-p5en | Qwen3-30B-A3B | 64 | p5en.48xlarge | bf16 | mfu | 33.7% | 28.6% | -15.2% | 28.6% | – | fit, layout unverified |
| mb2509-h100-dsv3 | DeepSeek-V3 | 1024 | dgx-h100 | fp8 | tokens_per_gpu | 1,297 tok/s/GPU | 1,455 tok/s/GPU | +12.2% | 21.4% | 8.7 / 7.8 | fit, layout unverified |
| mb2604-b200-dsv3 | DeepSeek-V3 | 256 | dgx-b200 | fp8 | tokens_per_gpu | 3,328 tok/s/GPU | 2,142 tok/s/GPU | -35.6% | 13.9% | 13.6 / 21.1 | hold-out |
| mb2604-b300-dsv3 | DeepSeek-V3 | 256 | dgx-b300 | fp8 | tokens_per_gpu | 3,456 tok/s/GPU | 3,722 tok/s/GPU | +7.7% | 24.1% | 13.1 / 12.1 | fit |
| mb2604-b200-qwen3-30b | Qwen3-30B-A3B | 8 | dgx-b200 | fp8 | tokens_per_gpu | 27,136 tok/s/GPU | 26,151 tok/s/GPU | -3.6% | 16.2% | 53.3 / 55.3 | fit |
| mb2604-b300-qwen3-30b | Qwen3-30B-A3B | 8 | dgx-b300 | fp8 | tokens_per_gpu | 30,720 tok/s/GPU | 28,920 tok/s/GPU | -5.9% | 17.9% | 47.1 / 50.0 | hold-out |
| mb2604-h100-qwen3-30b | Qwen3-30B-A3B | 16 | dgx-h100 | fp8 | tokens_per_gpu | 8,960 tok/s/GPU | 9,322 tok/s/GPU | +4.0% | 13.2% | 80.7 / 77.6 | hold-out |
| mb2604-b200-qwen3-235b | Qwen3-235B-A22B | 256 | dgx-b200 | fp8 | tokens_per_gpu | 3,776 tok/s/GPU | 5,220 tok/s/GPU | +38.2% | 19.4% | 12.0 / 8.7 | hold-out |
| mb2604-b300-qwen3-235b | Qwen3-235B-A22B | 256 | dgx-b300 | fp8 | tokens_per_gpu | 5,936 tok/s/GPU | 5,665 tok/s/GPU | -4.6% | 21.1% | 7.6 / 8.0 | hold-out |
| mb2604-h100-qwen3-235b | Qwen3-235B-A22B | 256 | dgx-h100 | fp8 | tokens_per_gpu | 1,712 tok/s/GPU | 1,613 tok/s/GPU | -5.8% | 13.6% | 26.4 / 28.0 | fit |
| repo-kimi-k2-b300-32n | Kimi-K2 | 256 | p6-b300.48xlarge | bf16 | tokens_per_gpu | 1,094 tok/s/GPU | 1,383 tok/s/GPU | +26.3% | 13.6% | 41.3 / 32.7 | hold-out, layout unverified |
| repo-dsv3-b300-32n | DeepSeek-V3 | 256 | p6-b300.48xlarge | bf16 | tokens_per_gpu | 1,067 tok/s/GPU | 1,386 tok/s/GPU | +29.9% | 17.9% | 42.4 / 32.6 | hold-out, layout unverified |

| Set | Runs | Median abs error | Max abs error |
|---|---|---|---|
| All scored | 38 | 6.0% | 38.2% |
| Hold-out | 23 | 7.4% | 38.2% |
| Fit | 15 | 5.7% | 15.2% |
| First set (29 runs, without the 2 superseded) | 27 | 5.7% | 17.8% |
| First set, hold-out | 16 | 7.3% | 17.8% |
| New MoE runs (mfu-model-3) | 11 | 7.7% | 38.2% |
| New MoE runs, hold-out | 7 | 26.3% | 38.2% |
| Superseded (not scored) | 2 | 79.8% | 113.0% |
| mfu-model-4 (before the B300 refit), all scored | 38 | 6.1% | 58.9% |
| mfu-model-1 on the first 29 runs | 29 | 5.6% | 31.0% |
| Old calculator (fixed 40% MFU), first 29 runs | 29 | 38.0% | 143.6% |

Days per 1T tokens = 1e12 / (tokens/s/GPU x GPUs) / 86400, from the reported and the predicted
throughput, so its error equals the throughput error. MFU-only runs (Meta, MT-NLG, Qwen3-30B-A3B)
publish no tokens/s; for those the MFU is compared directly. The only published wall time in the
set is DeepSeek-V3's (GPU-hours), compared with the 90% goodput applied.

Sources: [Megatron-Bridge performance archive](https://docs.nvidia.com/nemo/megatron-bridge/nightly/performance-summary-archive.html)
(25.09 and 26.04.01; MXFP8 counted as FP8); NeMo 25.07 and 25.04 performance summaries
([25.07](https://docs.nvidia.com/nemo-framework/user-guide/25.11/performance/archive/25.07-performance-summary.html),
[25.04](https://docs.nvidia.com/nemo-framework/user-guide/25.07/performance/archive/25.04-performance-summary.html));
[Llama 3 paper](https://arxiv.org/abs/2407.21783) Table 4; [DeepSeek-V3 paper](https://arxiv.org/abs/2412.19437)
(180K H800 GPU-hours per trillion tokens, 2048 GPUs); [MT-NLG paper](https://arxiv.org/abs/2201.11990) Sec. 2.4;
our runs under `benchmarks/pretrain-*` (paths in `validation_runs.py`).

Superseded runs: `nemo-h100-deepseek-v3` (887 tok/s/GPU, NeMo 25.07) and `nemo-b200-deepseek-v3` (1,640).
The same model on the same GPUs runs at 1,297 on H100 with NeMo 25.09 (+46%), and at 2,265 (25.09) and
3,328 (26.04.01) on B200, while dense rows barely moved in those releases (Llama 3 70B H100: 1,663,
1,619, 1,602). The MoE software changed, not the hardware, so these rows are kept for reference only.

Runs above 15%, and why:
- `mb2604-b200-qwen3-235b` +38% (+59% in mfu-model-4): the v0.4.2 recipe uses Megatron's `alltoall` token dispatcher (no
  DeepEP/HybridEP) for this row; the same model on B300 with HybridEP is -2%. The dispatcher is recorded
  (`plan["dispatcher"]`) but not modelled.
- `mb2604-b200-dsv3` -36% (-31% in mfu-model-4): DeepEP with all-to-all overlap (`moe_a2a_overlap`) hides more than the default
  EP overlap of 0.3; how much each dispatcher hides is not modelled. The Blackwell MoE half width lowers it 5 more points.
- `repo-dsv3-b300-32n` +30% and `repo-kimi-k2-b300-32n` +26% (+19% each in mfu-model-4): DeepEP over EFA, EP 32 over 4 nodes,
  TP 8, a dispatcher A/B layout. VP is 2, not the 1 of their config.yaml (the bench sets it; it lowers the bubble and adds about
  17 points). The measured all-to-all over EFA takes seconds a step where the model has 0.5 to 1.0 s (`docs/CALCULATOR-HARDWARE.md` 6b).
- `repo-qwen3-30b-a3b-bf16-p5en` -15%, `mb2509-h100-dsv3` +12%: fit runs;
  one expert-efficiency curve for 768- to 2,048-wide experts on Hopper.
- `deepseek-v3-paper` is -2.5% (+31% in mfu-model-1): 20 of 132 SMs run the all-to-all and RMSNorm / MLA
  up-projection are recomputed (paper Sec. 3.2.2, 3.2.3). The GPU-hours still include the batch-size
  warm-up and stalls beyond the 90% goodput.
- `nemo-b200-mixtral-8x22b` +18%: with the recipe's expert TP 1, EP 8 stays inside a node and the
  model's EP time is small; the measured run is slower.
- Llama 3 405B (Meta) +17%: VP and micro batch are not published; the run uses NeMo's VP 8. With VP 1
  the bubble is 32-48% and the model would under-predict instead.
- Nemotron-4 340B on B200 +14.7%: the B200 FP8 efficiency is fitted on Llama runs; very wide FFN layers
  on B200 run below it.

Families with no published plan and throughput we could find (not in the set): Grok-1, Qwen2.5/Qwen3
from the Qwen team, OPT-175B (FSDP plan and network not published in enough detail).

### 7a. Rack-scale hold-out: DGX GB200 (NVL72), NeMo 25.07

Added with the GB200 hardware step. 8 rows of the NeMo 25.07 performance summary
([source](https://docs.nvidia.com/nemo-framework/user-guide/25.11/performance/archive/25.07-performance-summary.html)), FP8, the first plan of each table row (the numbers in parentheses are other
settings). They are the same models and recipes as the DGX-B200 rows above, so they test what changes with a
72-GPU NVLink domain on the same Blackwell GPU. All are hold-out: no constant was fitted or changed for them
(`validation_runs.RACK_RUNS`, kept apart from the 38-run set so every number above is untouched).
`dgx-gb200` has 50 GB/s of network per GPU (4 x ConnectX-7 NDR 400 Gb/s per 4-GPU compute tray, [NVIDIA DGX
SuperPOD GB200 reference architecture](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/dgx-superpod-components.html)).
Not included: Nemotron5-H-56B, a hybrid Mamba-Transformer that the parser does not cover.

| Run | Model | GPUs | Hardware | Prec | Metric | Reported | Predicted | Error | Predicted MFU | Days per 1T tokens, reported / predicted | Set |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nemo-gb200-llama3-8b | Llama-3-8B | 8 | dgx-gb200 | fp8 | tokens_per_gpu | 31,737 tok/s/GPU | 35,962 tok/s/GPU | +13.3% | 41.6% | 45.6 / 40.2 | hold-out |
| nemo-gb200-llama3-70b | Llama-3-70B | 64 | dgx-gb200 | fp8 | tokens_per_gpu | 4,066 tok/s/GPU | 4,069 tok/s/GPU | +0.1% | 39.2% | 44.5 / 44.4 | hold-out |
| nemo-gb200-llama31-405b | Llama-3.1-405B | 128 | dgx-gb200 | fp8 | tokens_per_gpu | 706 tok/s/GPU | 730 tok/s/GPU | +3.5% | 38.4% | 128.1 / 123.8 | hold-out |
| nemo-gb200-llama4-scout | Llama-4-Scout | 64 | dgx-gb200 | fp8 | tokens_per_gpu | 13,783 tok/s/GPU | 14,407 tok/s/GPU | +4.5% | 34.9% | 13.1 / 12.6 | hold-out, layout unverified |
| nemo-gb200-llama4-maverick | Llama-4-Maverick | 128 | dgx-gb200 | fp8 | tokens_per_gpu | 11,682 tok/s/GPU | 15,587 tok/s/GPU | +33.4% | 37.7% | 7.7 / 5.8 | hold-out, layout unverified |
| nemo-gb200-mixtral-8x7b | Mixtral-8x7B | 64 | dgx-gb200 | fp8 | tokens_per_gpu | 17,246 tok/s/GPU | 17,900 tok/s/GPU | +3.8% | 29.7% | 10.5 / 10.1 | hold-out |
| nemo-gb200-mixtral-8x22b | Mixtral-8x22B | 256 | dgx-gb200 | fp8 | tokens_per_gpu | 2,869 tok/s/GPU | 3,375 tok/s/GPU | +17.6% | 34.0% | 15.8 / 13.4 | hold-out, layout unverified |
| nemo-gb200-deepseek-v3 | DeepSeek-V3 | 256 | dgx-gb200 | fp8 | tokens_per_gpu | 2,327 tok/s/GPU | 3,833 tok/s/GPU | +64.7% | 22.3% | 19.4 / 11.8 | hold-out, layout unverified |

Median absolute error 8.9%, worst 64.7% over 8 rows (10.4% and 82.3% in mfu-model-4; the bar is 15% typical,
30% worst). **Two rows miss the 30% bar and nothing was fitted on these rows:**
- `nemo-gb200-deepseek-v3` +65% (+82% in mfu-model-4): NeMo 25.07 software. The same pattern as the two superseded DeepSeek-V3 rows
  above (the same model on the same GPUs runs 2x faster in later releases); NeMo 26.08.01 publishes 4,912 tok/s/GPU
  for DeepSeek-V3 on 256 GB200 (MXFP8, PP 4, VP 4, EP 64). The dispatcher is assumed (`unverified`).
- `nemo-gb200-llama4-maverick` +33%: the token dispatcher and recompute settings are not recorded (`unverified`).

The same-GPU ratio the step is about: predicted GB200 / B200 against published for the same recipe is within
15% for Llama 3 70B and Mixtral 8x7B (`test_same_recipe_on_b200_and_gb200_keeps_its_ratio`).

### 7b. GB300 against GB200, same recipe (NeMo 26.04.01 and 26.08.01)

Added with the GB300 hardware step. 12 pairs of NeMo / Megatron-Bridge results
([26.08.01](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-summary.html), [26.04.01](https://docs.nvidia.com/nemo/megatron-bridge/nightly/performance-summary-archive.html)): the same model, global batch and sequence on DGX-GB300 and DGX-GB200, MXFP8 or FP8 counted
as FP8, BF16 for GPT-OSS 26.04.01. The check is the **ratio** GB300 / GB200: published against predicted, each system with its own
published plan. `dgx-gb300` has 100 GB/s of network per GPU (4 x ConnectX-8 800 Gb/s per compute tray, [NVIDIA DGX SuperPOD GB300
reference architecture](https://www.hostcomponents.com/wp-content/uploads/2026/03/dgx-spod-gb300-ra.pdf)).

**The cause of the gap and the term** (full analysis in `docs/CALCULATOR-HARDWARE.md` section 6): at the same listed dense FP8 peak and the same
HBM bandwidth, Blackwell Ultra's board power is up to 1,400 W against 1,200 W ([NVIDIA Table 2](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)),
and large-GEMM training runs at the power limit. Sustained tensor efficiency is scaled by
`(tgp_w / tgp_ref_w) ** TGP_EXPONENT`; the exponent is one constant, `TGP_EXPONENT = 1.25`, fitted on the two dense FP8 pairs (`fit_tgp_exponent`),
and applied to GB300 and (since mfu-model-5) B300. Before it, GB300 had the same efficiency as GB200 and the model predicted a ratio of about 1.0.
The "after" columns below are the GB300 step (mfu-model-4); mfu-model-5 moves no ratio by more than 1 point (full table further down).

Before and after (ratio GB300 / GB200):

| Pair | Set | Ratio reported | Ratio before | Error before | Ratio after | Error after |
|---|---|---|---|---|---|---|
| gb300-vs-gb200-llama3-70b-fp8-2604 | fit | 1.242 | 1.000 | -19.5% | 1.205 | -3.1% |
| gb300-vs-gb200-llama31-405b-fp8-2604 | fit | 1.185 | 1.001 | -15.6% | 1.199 | +1.2% |
| gb300-vs-gb200-llama3-70b-mxfp8-2604 | hold-out | 1.310 | 1.013 | -22.7% | 1.228 | -6.3% |
| gb300-vs-gb200-llama31-405b-mxfp8-2604 | hold-out | 1.200 | 1.033 | -13.9% | 1.247 | +3.9% |
| gb300-vs-gb200-dsv3-2604 | hold-out | 1.173 | 1.005 | -14.3% | 1.188 | +1.3% |
| gb300-vs-gb200-gpt-oss-120b-2604 | hold-out | 1.154 | 1.000 | -13.3% | 1.196 | +3.7% |
| gb300-vs-gb200-qwen3-30b-2604 | hold-out | 1.216 | 1.000 | -17.7% | 1.165 | -4.2% |
| gb300-vs-gb200-qwen3-235b-2604 | hold-out | 1.223 | 1.004 | -17.9% | 1.177 | -3.7% |
| gb300-vs-gb200-dsv3 | hold-out | 1.280 | 1.008 | -21.2% | 1.192 | -6.9% |
| gb300-vs-gb200-gpt-oss-120b | hold-out | 1.157 | 1.004 | -13.2% | 1.189 | +2.8% |
| gb300-vs-gb200-qwen3-30b | hold-out | 1.087 | 1.000 | -8.0% | 1.165 | +7.1% |
| gb300-vs-gb200-qwen3-235b | hold-out | 1.216 | 1.004 | -17.4% | 1.177 | -3.2% |

| Set | Pairs | Median error before | Worst before | Median error after | Worst after |
|---|---|---|---|---|---|
| Fit (dense FP8, 26.04.01) | 2 | 17.6% | 19.5% | 2.1% | 3.1% |
| **Hold-out, all** | 10 | 15.8% | 22.7% | **3.8%** | **7.1%** |
| Hold-out, dense MXFP8 | 2 | 18.3% | 22.7% | 5.1% | 6.3% |
| Hold-out, MoE, 26.04.01 | 4 | 16.0% | 17.9% | 3.7% | 4.2% |
| Hold-out, MoE, 26.08.01 | 4 | 15.3% | 21.2% | 5.1% | 7.1% |

The bar is 15% typical and 30% worst; both are met on every hold-out group. Full table, with the absolute errors of each system:

| Pair | GPUs | GB300 reported / predicted | GB200 reported / predicted | Ratio reported | Ratio predicted | Ratio error | GB300 error | GB200 error | Set |
|---|---|---|---|---|---|---|---|---|---|
| gb300-vs-gb200-llama3-70b-fp8-2604 | 64 | 5,248 / 5,054 | 4,224 / 4,196 | 1.242 | 1.205 | -3.1% | -3.7% | -0.7% | fit |
| gb300-vs-gb200-llama31-405b-fp8-2604 | 256 | 1,024 / 892 | 864 / 744 | 1.185 | 1.199 | +1.2% | -12.8% | -13.9% | fit |
| gb300-vs-gb200-llama3-70b-mxfp8-2604 | 64 | 4,864 / 5,054 | 3,712 / 4,115 | 1.310 | 1.228 | -6.3% | +3.9% | +10.8% | hold-out |
| gb300-vs-gb200-llama31-405b-mxfp8-2604 | 256 | 960 / 932 | 800 / 748 | 1.200 | 1.247 | +3.9% | -2.9% | -6.6% | hold-out |
| gb300-vs-gb200-dsv3-2604 | 256 | 4,992 / 4,801 | 4,256 / 4,031 | 1.173 | 1.191 | +1.5% | -3.8% | -5.3% | hold-out |
| gb300-vs-gb200-gpt-oss-120b-2604 | 64 | 19,200 / 19,668 | 16,640 / 16,411 | 1.154 | 1.198 | +3.9% | +2.4% | -1.4% | hold-out |
| gb300-vs-gb200-qwen3-30b-2604 | 8 | 31,744 / 33,516 | 26,112 / 28,588 | 1.216 | 1.172 | -3.6% | +5.6% | +9.5% | hold-out |
| gb300-vs-gb200-qwen3-235b-2604 | 256 | 6,944 / 6,786 | 5,680 / 5,739 | 1.223 | 1.182 | -3.3% | -2.3% | +1.0% | hold-out |
| gb300-vs-gb200-dsv3 | 256 | 6,288 / 4,818 | 4,912 / 4,031 | 1.280 | 1.195 | -6.6% | -23.4% | -17.9% | hold-out |
| gb300-vs-gb200-gpt-oss-120b | 64 | 33,024 / 37,066 | 28,544 / 31,109 | 1.157 | 1.191 | +3.0% | +12.2% | +9.0% | hold-out |
| gb300-vs-gb200-qwen3-30b | 8 | 44,544 / 33,516 | 40,960 / 28,588 | 1.087 | 1.172 | +7.8% | -24.8% | -30.2% | hold-out |
| gb300-vs-gb200-qwen3-235b | 256 | 8,832 / 6,786 | 7,264 / 5,739 | 1.216 | 1.182 | -2.7% | -23.2% | -21.0% | hold-out |

The absolute GB300 errors are larger than the ratio errors (for example -12.8% for 405B, -23.4% for DeepSeek-V3 26.08.01) because they carry the
errors the model already has on GB200 (-13.9% and -17.9% on the same rows): the plan, the dispatcher and the NeMo software release are not
pinned. No GB300 row is in the 38-run set. Hold-out after mfu-model-5: median 3.8%, worst 7.8% (`gb300-vs-gb200-qwen3-30b`, +7.1% before).

**B300 against B200, same plan** (NVIDIA's published runs; a hold-out check of the term on the B300; applied since mfu-model-5, exponent fitted on GB300 only):

Without the term (mfu-model-4) the model predicts a ratio of 1.0:

| Pair | Ratio reported | Ratio predicted | Ratio error |
|---|---|---|---|
| b300-vs-b200-dsv3 | 1.112 | 1.000 | -10.1% |
| b300-vs-b200-gpt-oss-120b | 1.123 | 1.000 | -10.9% |
| b300-vs-b200-qwen3-30b | 1.132 | 1.000 | -11.7% |

With 1,100 W against 1,000 W and the same exponent (mfu-model-5, shipped):

| Pair | Ratio reported | Ratio predicted | Ratio error |
|---|---|---|---|
| b300-vs-b200-dsv3 | 1.112 | 1.115 | +0.2% |
| b300-vs-b200-gpt-oss-120b | 1.123 | 1.120 | -0.2% |
| b300-vs-b200-qwen3-30b | 1.132 | 1.106 | -2.3% |

### 7c. AWS Trainium 1 (the only published Trainium training numbers)

Added with the Trainium hardware step. The Neuron documentation publishes one Trainium training table:
[Trn1/Trn1n Training Performance](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.27.1/about-neuron/benchmarks/trn1/trn1-training-performance.html), Neuron 2.24.0, NxD Training, BF16 autocast with an FP32 optimizer, micro batch 1,
global batch 1024, sequence 8192, 32 x trn1.32xlarge = 1,024 NeuronCores. Two rows (`validation_runs.TRN1_RUNS`): Llama-3.1-8B (47.95 seq/s, TP 32 DP 32)
and Llama-3.1-70B (7.94 seq/s, TP 32 DP 4 PP 8). A rank is a NeuronCore, so the throughput below is per NeuronCore.

One constant is fitted on them: `BASE_EFFICIENCY[("trainium1", "bf16")]` = 0.33 (`fit_trainium1`, geometric mean of the two per-row solutions).
**There is no hold-out: both rows are in the fit, and the errors below are in-sample.** Two rows from one source and one model family cannot also test the model.

| Run | Model | GPUs | Hardware | Prec | Metric | Reported | Predicted | Error | Predicted MFU | Days per 1T tokens, reported / predicted | Set |
|---|---|---|---|---|---|---|---|---|---|---|---|
| neuron-trn1-llama31-8b | Llama-3-8B | 1024 | trn1.32xlarge | bf16 | tokens_per_gpu | 384 tok/s/GPU | 435 tok/s/GPU | +13.5% | 26.5% | 29.5 / 26.0 | fit |
| neuron-trn1-llama31-70b | Llama-3-70B | 1024 | trn1.32xlarge | bf16 | tokens_per_gpu | 64 tok/s/GPU | 56 tok/s/GPU | -12.6% | 28.1% | 177.9 / 203.7 | fit |

In-sample error: median 13.1%, worst 13.5%. The model over-predicts the 8B run (+13.5%) and under-predicts the 70B run (-12.6%) with one constant:
the published MFU of the two runs is 23.4% and 32.2% (PaLM FLOPs, 190 TFLOPS per chip), a spread one constant cannot follow. They are not in the 38-run set; its summary is unchanged.

## 8. Plan search (`phase9_mfu_estimate.py`)

`candidate_plans`: TP in {1, 2, 4, 8} (<= GPUs per node), PP in the layer divisors plus {1, 2, 4, 8, 16}
(<= 16), CP 1 (CP up to 8 only for S >= 32K), EP in powers of 2 that divide the experts and GPUs / PP.
Fixed TP/PP/EP from the request restrict the list. All candidates are ranked by estimated tokens/s;
every one that fits in memory (phase-1 memory with that TP/PP/EP) is
scored with every ZeRO stage that fits (1, 2 and 3) and each of its viable micro batches; the fastest
wins, and the memory reported is that of the chosen stage. Before mfu-model-2 only the lowest stage
that fits was scored, and ZeRO-3 was timed with the ZeRO-2 formula, so ZeRO-3 plans (TP 1 PP 1) won
for large MoE models; and only the first 6 fitting plans were scored, which with ZeRO-3 timed
honestly are often the slow ones (scoring all of them takes < 0.1 s for 4,096 GPUs). An impossible fixed plan is an error. `max_fitting_plans` (default: all) caps how many fitting plans are scored.

## 9. Limits

- Activation memory is the reference calculator's. Recompute time is counted only when a plan lists
  `recompute` modules (the validation runs); the calculator's own plans assume none.
- Linear-attention FLOPs use 12 H d_lin^2 per layer (state update and read), not a kernel model.
- Network topology above the leaf (oversubscription) is not modelled.
- Hybrid Mamba models (Nemotron-H) are not covered by the parser (so Nemotron5-H-56B is not in section 7a).
