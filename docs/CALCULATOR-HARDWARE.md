# Calculator hardware: GB200 and GB300

Both calculators (portal Lambda `infra/lambda-live/calculator`, GitHub `ai-model-calculator`) read
the same hardware table, `mfu_model.HARDWARE`. This page lists every number that was added for rack-scale
systems and where it comes from. A figure that no AWS or NVIDIA page states is marked **unknown** and is
never replaced by a guess.

All BF16 and FP8 figures are dense (no sparsity), per GPU. "Derived" means computed here from the cited totals.

## 1. Three sizes that are all "8" on an HGX node

| Field | Meaning | HGX node (p5, p6-b200, p6-b300) | GB200 `u-p6e-gb200x72` | GB200 `u-p6e-gb200x36` |
|---|---|---|---|---|
| `domain_gpus` (`gpn`) | GPUs one NVLink domain reaches. TP, EP and CP groups up to this size stay on NVLink | 8 | 72 | 36 |
| `instance_gpus` | GPUs of the smallest launchable instance | 8 | 4 (`p6e-gb200.36xlarge`) | 4 |
| `billing_gpus` | GPUs bought as one unit (a Capacity Block buys whole UltraServers) | 8 | 72 | 36 |

The calculator sizes the UltraServer. A request for `p6e-gb200.36xlarge` or `GB200` means the 72-GPU
UltraServer. The default `gpus_per_node` comes from `billing_gpus`, so 8 GPUs of GB200 are billed as one
UltraServer (`billed_units` = 1 in the result).

## 2. GB200 (p6e-gb200)

| Field | Value | Source |
|---|---|---|
| GPUs per instance | p6e-gb200.36xlarge: 4 GPUs, Grace CPU (arm64), 144 vCPUs, 960 GiB host memory | [AWS instance table](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html) |
| Memory per GPU | 185 GiB (740 GiB / 4). The launch page's 13.4 TB / 72 = 186 GB is the same memory in other units. `hardware_config.py` moved from 186 to 185 | [AWS instance table](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html) |
| Dense BF16 | 2,500 TFLOPS (derived: 360 PF sparse / 2 / 72) | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| Dense FP8 | 5,000 TFLOPS (derived: 360 PF dense FP8 / 72) | [AWS P6e-GB200 launch](https://aws.amazon.com/about-aws/whats-new/2025/07/amazon-p6e-gb200-ultraservers-gpu-performance-ec2) |
| NVLink | 1.8 TB/s per GPU total, 900 GB/s per direction (130 TB/s / 72) | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| NVLink domain | 72 GPUs (u-p6e-gb200x72) or 36 (u-p6e-gb200x36) | [AWS P6e-GB200 launch](https://aws.amazon.com/about-aws/whats-new/2025/07/amazon-p6e-gb200-ultraservers-gpu-performance-ec2) |
| EFA per GPU | 100 GB/s (derived: 3200 Gigabit per instance / 4 GPUs = 800 Gbps). The same column gives p5/p5en/p6-b200 3200 Gbps / 8 = 50 GB/s and p6-b300 6400 / 8 = 100 GB/s, so one rule covers all hardware. Per UltraServer: 7,200 GB/s (x72), 3,600 (x36). The launch page's "up to 28.8 Tbps" fits the 36-GPU UltraServer (9 x 3.2 Tbps); AWS states neither on its pages | [AWS instance table](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html) |
| SMs per GPU | 148 (the B200 die) | `mfu_model.SMS_PER_GPU` |
| On-demand price | **Unknown.** Sold through EC2 Capacity Blocks for ML, price set at purchase. The result has `cost_estimate.price_known = false` and no cost line | [Capacity Blocks pricing](https://aws.amazon.com/ec2/capacityblocks/pricing/) |

If AWS confirms 28.8 Tbps for the x72 instead, cross-UltraServer data-parallel and expert-parallel time
doubles. The NeMo hold-out rows (CALCULATOR-MFU.md, section 7) use NVIDIA's own network, so they do not
check this number.

## 3. Rack-aware plan search (72-GPU domains)

72 is not a power of two, so a power-of-two layout (EP 64, for example) does not divide 72 GPUs. NeMo's GB200
recipes use 64, 128 or 256 GPUs, a part of each rack ([NeMo 25.07 performance summary](https://docs.nvidia.com/nemo-framework/user-guide/25.11/performance/archive/25.07-performance-summary.html)).
HGX nodes (domain 8) are not touched by any of this.

| Rule | Where |
|---|---|
| Automatic GPU counts: 4, 8, 16, 32, 64 (steps of `instance_gpus` inside one domain), then 72, 128, 144, 256, 288, ... (whole domains, and 64 per domain) | `mfu_model.auto_gpu_counts` |
| A request for n GPUs that is not a power of two is also tried as the largest power of two below it (72 -> 64, 144 -> 128). The faster plan that fits wins. The result has `rack_layout` with the GPUs requested, used and idle, and what was considered | Lambda `_run_once`, phase 9 `estimate_training` |
| n GPUs that do not fill whole domains (128 on 72-GPU UltraServers) run as 64 per UltraServer, so an EP, TP or CP group of up to 64 stays on NVLink | `mfu_model.rack_hw`, called by `estimate` |
| Billing is whole UltraServers: `ceil(requested GPUs / billing_gpus)`, so 64 of 72 bills 1, 128 bills 2. The idle GPUs are billed | `num_nodes`, `hardware.billed_units` |
| TP stays in {1, 2, 4, 8}. NeMo's GB200 runs use TP 1-2 and spend the domain on EP and PP | `mfu_model.candidate_plans` (unchanged) |

Not modelled: Grace CPU and NVLink-C2C. CPU offload over C2C is much faster than PCIe, so FSDP with offload on
GB200 is pessimistic.

## 4. Validation on GB200 (hold-out)

Eight DGX GB200 rows of the NeMo 25.07 performance summary, FP8, are scored as a hold-out (`validation_runs.RACK_RUNS`,
table in `docs/CALCULATOR-MFU.md` section 7a). Nothing was fitted to them. Median error 10.4%, worst +82.3%
(DeepSeek-V3, NeMo 25.07 software; layout unverified) and +35.3% (Llama 4 Maverick, layout unverified); the other six
rows are within +19.6%.

| Hardware | Network per GPU | Source |
|---|---|---|
| `dgx-gb200` | 50 GB/s (4 x ConnectX-7 NDR 400 Gb/s per 4-GPU compute tray) | [NVIDIA DGX SuperPOD GB200 reference architecture](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/dgx-superpod-components.html) |

The AWS GB200 EFA figure (100 GB/s per GPU) is not checked by these rows: they use NVIDIA's own network.

## 5. GB300 (P6e-GB300 UltraServer)

AWS has published little for P6e-GB300 ([GA announcement](https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-ec2-p6e-gb300-ultraservers-nvidia-gb300-nvl72-generally-available):
"contact your AWS sales representative"). What is published is used. paragao gave the network layout on 2026-11-17
(below, marked "paragao"). What nobody has published stays unknown, is listed in the result (`hardware.unknowns`) and is
never filled in by a guess; what paragao gave is listed in `hardware.user_provided`. The calculator's label is
`p6e-gb300-ultraserver`, not an AWS instance type: AWS has published no GB300 instance names.

| Field | Value | Source |
|---|---|---|
| Dense BF16 | 2,500 TFLOPS (derived: 360 PF with sparsity / 2 / 72). Same as GB200: the 1.5x uplift is FP4 only | [NVIDIA GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/), [AWS P6 page](https://aws.amazon.com/ec2/instance-types/p6/) ("1.5x ... FP4, without sparsity") |
| Dense FP8 | 5,000 TFLOPS (derived: 720 PF with sparsity / 2 / 72) | [NVIDIA GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/) |
| NVLink domain, bandwidth | 72 GPUs; 130 TB/s = 1.8 TB/s per GPU, 900 GB/s per direction | [NVIDIA GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/) |
| GPU memory | **279 GB per GPU is derived**: AWS says 1.5x the GPU memory of P6e-GB200 (13.4 TB), so 20.1 TB / 72. NVIDIA lists 20 TB per rack and 288 GB raw per GPU in partner datasheets. AWS gives no usable per-GPU figure | [AWS P6 page](https://aws.amazon.com/ec2/instance-types/p6/), [NVIDIA GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/) |
| SMs per GPU | 160 (the B300 die) | `mfu_model.SMS_PER_GPU` |
| EFA bandwidth per GPU | **100 GB/s, given by paragao, 2026-11-17** (not an AWS publication): one UltraServer is 72 B300 GPUs in one NVLink domain, made of 18 servers of 4 GPUs (+ 2 Grace CPUs each), every server with 8 EFA adapters at 400 Gbps = 3200 Gbps = 800 Gbps per GPU = 100 GB/s per GPU, the same per-GPU figure as GB200. EFA carries only traffic **between** UltraServers; inside the 72-GPU domain everything is NVLink. `p6e-gb300` is not in the [AWS instance table](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html) | paragao, 2026-11-17 |
| Instance types and sizes | **Unknown** as AWS instance names. 4 GPUs per server is from paragao, 2026-11-17 (and the NVL72 compute tray, below), used for the GPU-count steps | [NVIDIA DGX SuperPOD GB300 reference architecture](https://www.hostcomponents.com/wp-content/uploads/2026/03/dgx-spod-gb300-ra.pdf), paragao, 2026-11-17 |
| On-demand price | **Unknown.** No cost line | none |

Validation: twelve GB300 / GB200 pairs of NeMo / Megatron-Bridge results (`docs/CALCULATOR-MFU.md` section 7b). GB300 runs faster than GB200 at the same listed peak;
section 6 gives the cause and the term. Ratio error on the 10 hold-out pairs: median 3.8%, worst 7.1% (before the term: 15.8% and 22.7%).
`dgx-gb300` (validation only) has 100 GB/s per GPU (4 x ConnectX-8 800 Gb/s per 4-GPU compute tray).

### 5a. Scale-out: 72 and 144 GPUs (GB200 x72 and GB300 are the same)

The 72-GPU limit for GB300 is gone: it scales out by whole UltraServers, like GB200. What the model does at the NVLink
domain boundary (`mfu_model._span_bw`, `estimate`):

| GPUs | Which links | Test |
|---|---|---|
| 72 (one UltraServer) | TP, PP, EP and DP traffic all use NVLink. Making the EFA 10^8 times slower changes no time | `test_72_gpus_all_comm_on_nvlink_zero_efa_time` |
| 144 (two UltraServers) | Groups inside one UltraServer (TP 2, EP 72) use NVLink only. A group that spans both uses the EFA: EP 144 (half of the all-to-all leaves the UltraServer) and the ZeRO-3 / DP group of 144 | `test_144_gpus_efa_only_for_the_groups_that_span_both_ultraservers` |

A group spread over both UltraServers is limited by the smaller of NVLink and the EFA of the ranks it has in each UltraServer
(`min(NVLink, ranks per UltraServer x 100 GB/s x 0.8)`). With 72 ranks per UltraServer that is 5,760 GB/s of EFA against 684 GB/s of NVLink,
so a DP group that uses all 72 GPUs of each UltraServer stays NVLink-limited, and only groups with few ranks per UltraServer
(or the EP all-to-all, which is timed on the EFA directly) are EFA-limited. This is the same rule the model uses for HGX nodes.

Example, Llama-3.1-70B BF16 on GB300 (with the board-power term of section 6), 2T tokens, calculator Lambda (plan search chooses TP/PP/ZeRO):

| GPUs | UltraServers | Plan | MFU | tokens/s | tokens/s per GPU | Days |
|---|---|---|---|---|---|---|
| 72 | 1 | TP 1, PP 4, DP 18, ZeRO-1 | 53.0% | 212,271 | 2,948 | 121.2 |
| 144 | 2 | TP 1, PP 1, DP 144, ZeRO-2 | 52.9% | 423,621 | 2,942 | 60.7 |

Example, mfu_model, DeepSeek-V3 shape with 288 experts, FP8, EP = number of GPUs, GB300:

| GPUs | EP all-to-all per step | Step | MFU |
|---|---|---|---|
| 72 | 314.2 ms (NVLink) | 1.546 s | 30.8% |
| 144 | 1,352.8 ms (half over EFA) | 2.273 s | 21.0% |

## 6. Why GB300 is faster than GB200 at the same peak

Observed: in the 12 NeMo pairs the same recipe runs 8.7% to 31% faster per GPU on GB300 (dense FP8 70B with the same plan and micro batch on both:
+24%; 405B: +19%). The model gave GB300 the same compute efficiency as GB200 and so predicted a ratio of 1.0 (typical error 15.8%, worst 22.7% on the hold-out).

Candidate causes, with the numbers that rule them in or out (NVIDIA's [Blackwell Ultra table](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/) unless marked):

| Cause | Blackwell (GB200) | Blackwell Ultra (GB300) | Explains the gap? |
|---|---|---|---|
| Dense FP8 peak | 5 PFLOPS | 5 PFLOPS | No: equal. The 1.5x of Blackwell Ultra is NVFP4 only (15 against 10 PFLOPS); the NVFP4 pairs show about 1.5x, as expected, and are not modelled |
| HBM bandwidth | 8 TB/s | 8 TB/s | No: equal |
| HBM capacity | 192 GB | 288 GB | No: the 70B FP8 pair uses the same plan and micro batch 2 on both and still shows +24%. Larger micro batches matter only where the published plans differ (the plan search already follows memory) |
| Attention softmax (SFU exponentials) | 5.0 tera-exp/s | 10.7 tera-exp/s | Not more than 1 to 2 points: the exponentials of attention take 3.5% (70B, 8k) and 2.3% (405B, 8k) of a GB200 second even at the SFU peak (`attention_exp_share`), so a 2.14x faster SFU saves under 2% of 18% to 24% |
| Board power (TGP) | up to 1,200 W | up to 1,400 W | **Yes, mostly.** Ratio 1.167. Large-GEMM transformer training draws the whole envelope (ViT-Large: 92 to 97% of TDP on both B200 and B300; B300 +16% training and +21% inference at 1,100 W against 1,000 W, [arXiv 2610.41919](https://arxiv.org/abs/2610.41919)). At a fixed listed peak, sustained tensor throughput follows the power headroom |
| Software release | | | Noise of about +-10% in any ratio: between NeMo 26.04.01 and 26.08.01 GB300 gained 26% on DeepSeek-V3 and 40% on Qwen3-30B-A3B, GB200 15% and 57%. The dense pairs of 26.04.01 and the MoE pairs of both releases give the same ratio band |

The term is `eta_base = BASE_EFFICIENCY[family, precision] x (tgp_w / tgp_ref_w) ** TGP_EXPONENT` (`mfu_model.estimate`), a general term for any system
that has a board power and the board power of the Blackwell part whose constant it uses: GB300 has 1,400 W and 1,200 W, and since mfu-model-5 the
HGX B300 has 1,100 W and 1,000 W (section 6a). The one constant, `TGP_EXPONENT = 1.25`, is
fitted on two pairs and checked on ten. An exponent above 1 says the power ratio (1.167) accounts for about 1.17 of the 1.21 gain, and about 3 to 4 points of the gap come
from somewhere this data cannot separate (performance per watt, kernels tuned for sm_103). They are not hidden: they sit in the exponent.

Before and after on the hold-out (ratio error GB300 / GB200): median 15.8% to 3.8%, worst 22.7% to 7.1% (`docs/CALCULATOR-MFU.md` section 7b).

### 6a. HGX B300: the term is applied, with a joint refit (mfu-model-5)

HGX B300 against HGX B200 at the same plan is 1.11, 1.12 and 1.13 times faster (Megatron-Bridge 26.04 and 26.04.01: DeepSeek-V3, GPT-OSS, Qwen3-30B-A3B), at the same
listed dense peak (2,250 / 4,500 TFLOPS). The board power differs: B200 is configurable up to 1,000 W per GPU
([NVIDIA HGX B200 datasheet](https://images.nvidia.com/aem-dam/Solutions/documents/HGX-B200-PCF-Summary.pdf)) and the B300 runs a 1,100 W envelope
([arXiv 2610.41919](https://arxiv.org/abs/2610.41919); [Supermicro HGX B300](https://learn-more.supermicro.com/b300): "eight 1,100W TDP Blackwell GPUs").
No NVIDIA or AWS page found gives the HGX B300 per-GPU TDP, and AWS does not publish the p6-b300 power setting: 1,100 W is assumed for `p6-b300.48xlarge`.
`p6-b300.48xlarge` and `dgx-b300` now carry `tgp_w` 1100 and `tgp_ref_w` 1000: a gain of 1.1 ^ 1.25 = 1.127 on the base efficiency.

**The joint refit.** Adding the term alone would push the B300 rows that the Blackwell constants were fitted on 9 to 12 points up. What was refitted, in
the order of `validation_runs.fit`:

| Constant | mfu-model-4 | mfu-model-5 | Fitted on |
|---|---|---|---|
| BASE_EFFICIENCY blackwell / bf16 | 0.443 | **0.417** | the 2 dense Blackwell BF16 fit rows (our Llama 70B on p6-b200, our Qwen3-8B on p6-b300); the B300 row now carries x1.127 |
| BASE_EFFICIENCY blackwell / fp8, hopper, ampere | 0.417; 0.573, 0.45; 0.471 | unchanged | the refit gives the same values (their fit rows have no B300) |
| TGP_EXPONENT | 1.25 | 1.25 | the 2 GB300 fit pairs (ratios; the refit gives 1.25 again) |
| MOE_EXPERT_FACTOR, MOE_EXPERT_HALF_FFN, OVERLAP ep | 0.78, 1024, 0.3 | unchanged (kept) | the mfu-model-4 grid on the 6 MoE fit rows (`fit_moe_shared` on `mfu_model_4()` reproduces it) |
| MOE_EXPERT_HALF_FFN_FAMILY blackwell | (1024, shared) | **1664** | the 2 Blackwell MoE fit rows (mb2604-b300-dsv3, mb2604-b200-qwen3-30b), shared constants held |

Why a Blackwell MoE half width: with the B300 term the Blackwell MoE rows ran high on both GPUs (mb2604-b300-dsv3 +18%, mb2604-b200-qwen3-30b +13%, mb2604-b300-qwen3-30b
+10%; the B200 row was already +15% before). The shared MoE curve had been pulled up by the B300 rows getting no power gain. One more constant, the expert width at
which a routed-expert GEMM reaches half of MOE_EXPERT_FACTOR on Blackwell, fits both Blackwell fit rows. It is checked on rows it never saw: the 16 absolute
GB300 and GB200 MoE rows of the 8 MoE pairs (median 15.4% to 9.2%) and the GB200 rack rows (median 10.4% to 8.9%).

**The alternative that was not taken:** refit the shared MoE constants together with the Blackwell width (`fit_moe_joint`: 0.79, 768, 0.3, Blackwell 1728). It lowers
the fit-row median (5.7% to 4.4%) but raises the hold-out median (7.4% to 7.8%) and the 38-run median (6.0% to 6.4%), and it moves every Hopper MoE answer
(expert matmuls about 10% faster at 2,048-wide experts; `repo-qwen3-30b-a3b-bf16-p5en` -15.2% to -6.9%, `mb2604-h100-qwen3-30b` +4.0% to +9.8%). Nothing in the
B300 change needs it, so Hopper and p5en are untouched (`test_hopper_and_ampere_rows_do_not_move`, the p5en pins, parity 11 of 11).

**Fit and hold-out sets.** No row moved between the sets. The 15 fit rows and 23 hold-out rows of the 38-run set are the same as in mfu-model-4; the 3 B300 / B200
pairs stay a hold-out (the exponent is fitted on GB300 only), the GB300 pairs keep their 2 fit and 10 hold-out, the GB200 rack rows stay hold-out.

**Results against the targets:**

| Target | Bar | mfu-model-4 | mfu-model-5 |
|---|---|---|---|
| 38-run median | within about 1 point of 6.1% | 6.1% | **6.0%** (hold-out 7.4% and 7.4%) |
| 38-run worst | no new row past 30% unless explained | 58.9% | 38.2%; no row newly past 30% (the two past 30% were past it before) |
| GB300 / GB200 pairs, hold-out | under 15% typical | 3.8% (worst 7.1%) | **3.8%** (worst 7.8%) |
| B300 / B200 published ratios | within 5% | -10.1%, -10.9%, -11.7% | **+0.2%, -0.2%, -2.3%** |
| Our p6-b300 rows | within 15% or explained | -5.0%, +19.0%, +18.7% | **+0.7%** (Qwen3-8B dense); +26.3% and +29.9% for the two MoE dispatcher runs, explained in 6b |

**Every row that moves more than 2 points** (error against published, mfu-model-4 to mfu-model-5):

| Row | Set | Before | After | Why it moves |
|---|---|---|---|---|
| repo-llama70b-bf16-b200 (ours, p6-b200) | fit | +5.3% | -0.7% | Blackwell BF16 base 0.443 to 0.417 |
| repo-qwen3-8b-bf16-b300 (ours, p6-b300) | fit | -5.0% | +0.7% | power term x1.127 and the new base |
| nemo-b200-mixtral-8x7b | hold-out | +2.0% | -1.4% | Blackwell MoE half width |
| mb2604-b200-dsv3 | hold-out | -30.8% | -35.6% | Blackwell MoE half width; already past 30% (dispatcher overlap not modelled, `EXPLAINED_OUTLIERS`) |
| mb2604-b200-qwen3-30b | fit | +15.0% | -3.6% | Blackwell MoE half width |
| mb2604-b300-qwen3-30b | hold-out | +1.6% | -5.9% | power term, then the half width |
| mb2604-b200-qwen3-235b | hold-out | +58.9% | +38.2% | Blackwell MoE half width; already past 30% (alltoall dispatcher) |
| mb2604-b300-qwen3-235b | hold-out | -1.2% | -4.6% | power term, then the half width |
| repo-kimi-k2-b300-32n (ours, p6-b300) | hold-out | +19.0% | +26.3% | VP corrected 1 to 2 (+14 points), power term, half width; section 6b |
| repo-dsv3-b300-32n (ours, p6-b300) | hold-out | +18.7% | +29.9% | VP corrected 1 to 2 (+17 points), power term, half width; section 6b |
| nemo-b200-deepseek-v3 | superseded | +136.3% | +113.0% | not scored |
| nemo-gb200-llama4-scout (rack) | hold-out | +7.2% | +4.5% | Blackwell MoE half width |
| nemo-gb200-mixtral-8x7b (rack) | hold-out | +7.4% | +3.8% | Blackwell MoE half width |
| nemo-gb200-deepseek-v3 (rack) | hold-out | +82.3% | +64.7% | Blackwell MoE half width; NeMo 25.07 software |

`mb2604-b300-dsv3` (fit) moves +7.4% to +7.7%. No GB300 / GB200 ratio moves by more than 1 point (both sides are Blackwell). Every Hopper and Ampere row is unchanged.

**Live answers** (Lambda, 20T tokens, sequence 4096, BF16; auto, a fixed GPU count, and a 30-day target), mfu-model-4 to mfu-model-5:

| Case | GPUs | TP/PP/DP/EP | MFU before | MFU after | tok/s before | tok/s after | days before | days after | cost USD before | cost USD after | change |
|---|---|---|---|---|---|---|---|---|---|---|---|
| glm-p5en-auto | 16 | 4/4/1/4 | 0.3661 | 0.3661 | 49,419 | 49,419 | 5204.4 | 5204.4 | 34,973,707 | 34,973,707 | same |
| glm-p5en-64 | 64 | 2/16/2/1 | 0.3949 | 0.3949 | 213,197 | 213,197 | 1206.4 | 1206.4 | 32,427,986 | 32,427,986 | same |
| glm-p5en-30d | 4096 | 8/16/32/1 | 0.3254 | 0.3254 | 11,245,265 | 11,245,265 | 22.9 | 22.9 | 39,347,149 | 39,347,149 | same |
| glm-b200-auto | 8 | 8/1/1/8 | 0.2715 | 0.2315 | 41,680 | 35,538 | 6170.7 | 7237.3 | 26,657,544 | 31,265,098 | -14.7% tok/s |
| glm-b200-64 | 64 | 1/16/4/1 | 0.3075 | 0.2565 | 377,660 | 315,077 | 681.0 | 816.3 | 23,536,710 | 28,211,758 | -16.6% tok/s |
| glm-b200-30d | 2048 | 1/1/2048/8 | 0.2775 | 0.2358 | 10,906,048 | 9,268,642 | 23.6 | 27.7 | 26,081,347 | 30,688,901 | -15.0% tok/s |
| glm-b300-auto | 8 | 4/2/1/4 | 0.2864 | 0.2705 | 43,969 | 41,528 | 5849.5 | 6193.4 | 35,097,121 | 37,160,441 | -5.6% tok/s |
| glm-b300-64 | 64 | 1/16/4/1 | 0.3075 | 0.289 | 377,660 | 354,935 | 681.0 | 724.6 | 32,689,875 | 34,782,907 | -6.0% tok/s |
| glm-b300-30d | 2048 | 1/1/2048/8 | 0.2887 | 0.2725 | 11,346,820 | 10,712,152 | 22.7 | 24.0 | 34,816,955 | 36,879,770 | -5.6% tok/s |
| glm-gb300-auto | 8 | 4/2/1/4 | 0.3376 | 0.2868 | 57,597 | 48,922 | 4465.5 | 5257.4 | unknown | unknown | -15.1% tok/s |
| glm-gb300-72 | 72 | 1/9/8/1 | 0.3723 | 0.3106 | 571,601 | 476,913 | 450.0 | 539.3 | unknown | unknown | -16.6% tok/s |
| glm-gb300-30d | 2048 | 1/1/2048/16 | 0.346 | 0.2928 | 15,112,199 | 12,788,502 | 17.0 | 20.1 | unknown | unknown | -15.4% tok/s |
| param2-p5en-auto | 8 | 1/2/4/1 | 0.3869 | 0.3869 | 175,504 | 175,504 | 1465.5 | 1465.5 | 4,924,073 | 4,924,073 | same |
| param2-p5en-64 | 64 | 1/1/64/1 | 0.3866 | 0.3866 | 1,403,038 | 1,403,038 | 183.3 | 183.3 | 4,927,578 | 4,927,578 | same |
| param2-p5en-30d | 512 | 1/1/512/1 | 0.3818 | 0.3818 | 11,085,193 | 11,085,193 | 23.2 | 23.2 | 4,989,416 | 4,989,416 | same |
| param2-b200-auto | 8 | 1/1/8/1 | 0.2994 | 0.2471 | 308,966 | 254,978 | 832.5 | 1008.7 | 3,596,216 | 4,357,673 | -17.5% tok/s |
| param2-b200-64 | 64 | 1/1/64/1 | 0.2985 | 0.2465 | 2,464,470 | 2,034,875 | 104.4 | 126.4 | 3,606,815 | 4,368,272 | -17.4% tok/s |
| param2-b200-30d | 256 -> 512 | 1/1/256/1 -> 1/1/512/1 | 0.2957 | 0.2421 | 9,765,624 | 15,988,006 | 26.3 | 16.1 | 3,640,889 | 4,447,778 | +63.7% tok/s |
| param2-b300-auto | 8 | 1/1/8/1 | 0.2994 | 0.2783 | 308,966 | 287,234 | 832.5 | 895.4 | 4,994,744 | 5,372,651 | -7.0% tok/s |
| param2-b300-64 | 64 | 1/1/64/1 | 0.299 | 0.278 | 2,468,295 | 2,294,903 | 104.2 | 112.1 | 5,001,701 | 5,379,608 | -7.0% tok/s |
| param2-b300-30d | 256 | 1/1/256/1 | 0.2976 | 0.2768 | 9,826,694 | 9,139,412 | 26.2 | 28.1 | 5,025,364 | 5,403,270 | -7.0% tok/s |
| param2-gb300-auto | 4 | 1/1/4/1 | 0.363 | 0.2996 | 208,138 | 171,766 | 1235.7 | 1497.4 | unknown | unknown | -17.5% tok/s |
| param2-gb300-72 | 72 | 1/1/72/1 | 0.3623 | 0.2991 | 3,738,601 | 3,086,419 | 68.8 | 83.3 | unknown | unknown | -17.4% tok/s |
| param2-gb300-30d | 256 | 1/1/256/1 | 0.3602 | 0.2977 | 13,217,257 | 10,922,396 | 19.5 | 23.5 | unknown | unknown | -17.4% tok/s |

p5en: identical (no Hopper constant changed). B200: 15% to 18% fewer tokens/s for both MoE models: the Blackwell BF16 base (-5.9%) and the Blackwell MoE half width
(-17% on the expert matmuls of 2,048-wide experts). B300: 6% to 7% fewer: the same two, less the power gain (+12.7%). GB300: 15% to 18% fewer: the half width and the base
(GB300 already had its power term). Param2 on B200 with a 30-day target now needs 512 GPUs instead of 256. These are the moves the B300 and B200 MoE rows asked for;
dense answers on B300 rise (Llama 3 70B BF16 at 128 GPUs: +6%).

### 6b. Our p6-b300 MoE rows: why the model is high

The two rows are DeepEP dispatcher A/B runs, not tuned training runs: TP 8, PP 8, EP 32, DP 4 on 32 p6-b300, BF16, micro batch 4, global batch 256, mock data with
forced load balancing ([DeepSeek-V3 results](https://github.com/awslabs/awsome-distributed-ai/blob/main/examples/training/megatron-bridge/dsv3/benchmarks/RESULTS.md),
[Kimi-K2 results](https://github.com/awslabs/awsome-distributed-ai/blob/main/examples/training/megatron-bridge/kimi-k2/benchmarks/RESULTS.md); the repo pages say
"Data sourced from awsome-distributed-ai"). What was checked against the published B300 / B200 pairs:

| Item | Our rows | Published B300 / B200 pairs | Finding |
|---|---|---|---|
| Precision | BF16 | MXFP8 (DeepSeek-V3, Qwen3-30B), BF16 (GPT-OSS) | modelled per precision; not the cause |
| Plan | TP 8, PP 8, **VP 2**, EP 32 over 4 nodes, ETP 1 | TP 1, EP 8 inside one node (all-to-all on NVLink only) | ours sends 75% of the all-to-all over EFA |
| VP | config.yaml says 1 | | **mislabelled**: with overlap on and PP > 1 the bench sets VP 2 (the recipe's (8, 2) layout) and turns recompute off (`bench_kimi_k2_pretrain.py`, `bench_dsv3_pretrain.py`; RESULTS.md: "overlap=on required VPP=2 + recompute OFF"). Corrected in `validation_runs.py`; it lowers the modelled bubble from 30% to 18% |
| Recompute | none recorded | per recipe | none (granularity None in the overlap-on bench): matches the row |
| Software | config.yaml: CUDA 12.8, NCCL 2.25+ | NeMo 26.04 / 26.04.01 | the images are `megatron-bridge-uccl:nemo-26.04.01-...` (Bridge 0.4.2, Core 0.17.1), the same release as the pairs; the NVSHMEM image is a `-cu13` build. The CUDA line of config.yaml is a template value |
| GPU, network | config.yaml: "B300 192 GB", "EFA v2 (3200 Gbps)" | | **mislabelled** template values: p6-b300 has 288 GB per GPU and 6,400 Gbps (the RESULTS pages log 16 EFA NICs per node, `efa-direct`). The model uses 100 GB/s per GPU |
| Nodes | 32 | 1 to 32 | right |
| Power cap, clocks | not logged | not published | unknown |
| Layout imbalance | the (8, 2) layout puts 8 decoder layers on 7 of the 8 stages (61 / 8 = 7.6 average) | | about 5% the model does not count (it counts the average stage) |

The cause is the all-to-all over EFA. The same bench with overlap off, changing only the dispatcher (NCCL alltoall to DeepEP), moves the step by 3.5 s (DeepSeek-V3:
9.77 to 6.26 s) and 3.3 s (Kimi-K2: 9.30 to 6.04 s), so the EP all-to-all costs seconds a step on this layout. The model's whole EP time is 0.5 s (DeepSeek-V3) and 1.0 s
(Kimi-K2), and its overlap-off step is 2.4 s and 2.1 s shorter than the measured DeepEP step (`test_our_b300_moe_rows_are_high_because_of_the_efa_all_to_all`). The
published B300 / B200 pairs keep EP inside one NVLink node, so the power term and the Blackwell constants are tested where the EFA does not matter, and the EP-over-EFA
gap shows only on our rows. Fixing it means a measured EFA all-to-all efficiency for DeepEP, which also moves p5en MoE answers (EP across nodes): not part of this change.
The portal repo's `benchmarks/pretrain-*-moe-bf16/p6-b300-32nodes/config.yaml` files are left as they are; the mislabels above are listed for a separate fix.
S3 `repos/benchmarks/` is the CI copy of that folder (portal `docs/CI.md`) and was not read: this host has no access to the portal account.

## 7. AWS Trainium: Trn1 (calibrated), Trn2 and Trn3 (uncalibrated)

Sources are the Neuron pages [Trainium](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/trainium.html),
[Trn1](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/trn1-arch.html),
[Trainium2](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html),
[Trn2](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn2-arch.html),
[Trainium3](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium3.html),
[Trn3](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn3-arch.html) and the AWS
[instance table](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html). Each number is linked in the code comments of `mfu_model.HARDWARE`.

### 7.1 Chips and NeuronCores

The request and the result count **chips** (the `num_gpus` field). The plan counts **ranks = logical NeuronCores**, as the Neuron recipes do (Llama 3.1 TP 32 = 32 cores = 16 chips of one
trn1.32xlarge). TP, PP and DP degrees in the result are in NeuronCores; `hardware.chips`, `hardware.ranks` and `hardware.ranks_per_chip` say how they relate.
Memory per GPU in the result is memory per NeuronCore.

| System | Chips per instance | NeuronCores per chip | BF16 per NeuronCore | Memory per NeuronCore | Price per instance hour |
|---|---|---|---|---|---|
| trn1.32xlarge | 16 | 2 (NeuronCore-v2) | 95 TFLOPS (190 per chip) | 16 GiB (32 GiB per chip) | 21.50 USD ([AWS Trn1 page](https://aws.amazon.com/ec2/instance-types/trn1/), region not stated) |
| trn1n.32xlarge | 16 | 2 | 95 TFLOPS | 16 GiB | 24.78 USD (same page) |

Link tiers (`mfu_model._tier_span_bw`): one NeuronLink-v2 tier (2D torus, 384 GB/s per chip, 96 GB/s per NeuronCore per direction, 32 NeuronCores = one instance), then the EFA:
800 Gbps per instance on trn1 (3.125 GB/s per NeuronCore) and 1600 Gbps on trn1n (6.25). A group that spans instances is limited by the smaller of the torus and its NeuronCores' EFA share,
the rule the GPUs use. The link efficiencies of the torus and the switch (`LINK_EFFICIENCY`) are assumptions equal to NVLink's; no Neuron measurement exists.

### 7.2 Neuron rules, each tied to a Neuron SDK version

| Rule in the calculator | Neuron SDK version and source |
|---|---|
| Plans are DP, TP (up to 32 NeuronCores), PP and ZeRO-1 only | NxD Training (NxDT) supports DP, TP, sequence parallel, PP (1F1B, interleaved) and ZeRO-1, [NxDT overview, 2.27.1](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.27.1/libraries/nxd-training/overview.html); the two calibration rows are NxDT, 2.24.0 |
| NxDT is end of support from Neuron 2.29 (April 2026), maintenance only in 2.30, not in the DLAMIs from 2.31; the native PyTorch path (DDP, FSDP2, DTensor TP) replaces it | [What's new, 2.32.0](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/about-neuron/whats-new.html). The rules above may go stale; they describe the 2.24 recipes the data comes from |
| EP and CP are not planned | MoE training collectives are experimental in Neuron 2.31 (same page). A MoE model is sized without EP and is labelled uncalibrated for the experts |
| FP8 / MXFP8 is refused (error, not a warning) | MXFP8 training kernels (2.30) and MXFP8 blockwise-MoE forward (2.32) are experimental kernels in the NKI Library; no end-to-end FP8 pre-training recipe (same page) |
| Memory mode `bf16_fp32_optimizer` (default) | BF16 autocast with an FP32 optimizer, the mode of the published rows: the reference mixed-precision memory accounting |
| Memory mode `bf16_sr` (request field `neuron_precision`) | BF16 with stochastic rounding, no FP32 master copy ([AWS blog, Llama 2 7B on 128 trn1](https://aws.amazon.com/blogs/machine-learning/end-to-end-llm-training-on-instance-clusters-with-over-100-nodes-using-aws-trainium/)): weights 2 B, gradients 2 B, optimizer states 4 B. The 2 + 4 bytes are an assumption: no Neuron page states them |

Not modelled: the XLA compile time and graph cache, a one-time cost per job.

### 7.3 Trn1 calibration

One constant fitted on the 2 published rows, no hold-out (`docs/CALCULATOR-MFU.md` section 7c): in-sample median error 13.1%, worst 13.5%. The
published TP 32 PP 8 plan of Llama-3.1-70B is viable in the memory model, and its estimate is within 15% of the published run. Every Trn1 result carries the calibration text
in `hardware.calibration` and in `disclaimer`, which the reports print. Unknown: the usable HBM per NeuronCore after the Neuron runtime reserve (the memory model uses the full 16 GiB).

### 7.4 Trn2 (uncalibrated)

**Label on every Trn2 result: "uncalibrated: no published Trn2 training data".** The Neuron documentation publishes a Trn1 training table only
([benchmarks index](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.28.1/about-neuron/benchmarks/index.html)); no Trn2 training throughput or MFU was found.
The result says it in `hardware.calibration` and in `disclaimer` (which the reports print), the agent repeats it, and the price is unknown.
The estimate is a model with Trn1's base efficiency (`BASE_EFFICIENCY[("trainium2", "bf16")]`, an assumption, pinned equal to Trn1's by a test)
and the same link efficiencies as NVLink (assumption). To calibrate Trn2 it must be measured (for example Llama 3.1 8B and 70B BF16 on 2 to 8 trn2.48xlarge).

| Field | Value | Source |
|---|---|---|
| Chip | 8 physical NeuronCore-v3; 667 BF16 TFLOPS dense; 1,299 cFP8 TFLOPS (not used: FP8 training is refused, Neuron 2.30 and 2.32); 96 GiB HBM, 2.9 TB/s | [Trainium2](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html) |
| Rank | logical NeuronCore, LNC=2 (the default): 4 per chip, 64 per trn2.48xlarge; 166.75 BF16 TFLOPS and 24 GiB each. LNC=1 (8 per chip) is not modelled | [LNC](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-features/logical-neuroncore-config.html) |
| trn2.48xlarge | 16 chips in a 4x4 2D torus, NeuronLink-v3 1,024 GiB/s per chip (tier 1, 137.44 GB/s per NeuronCore per direction); EFAv3 3.2 Tbps = 400 GB/s per instance = 25 GB/s per chip. The instance page prints "3,200 GB/s" next to "3.2 Tbps": a unit slip for Gbps | [Trn2 instances](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn2-arch.html), [AWS instance table](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html) |
| Trn2 UltraServer (`trn2-ultraserver`, also `trn2u`, `trn2u.48xlarge`) | 4 x trn2u.48xlarge = 64 chips; chips with the same coordinates form a ring across the instances, 256 GiB/s per chip (tier 2, 34.36 GB/s per NeuronCore per direction); billed as a whole UltraServer of 64 chips | same page |
| Not modelled | trn2.3xlarge (1 chip, no NeuronLink, no EFA); LNC=1 | same page |
| Price | **Unknown.** No AWS or Neuron page checked lists an on-demand price. A third-party tracker lists one; it is not used | none |

Three link tiers: 16 chips (64 NeuronCores) at 1,024 GiB/s per chip, the 64-chip UltraServer (256 NeuronCores) at 256 GiB/s per chip, and the EFA at 25 GB/s per chip
between UltraServers (`test_group_bandwidth_follows_the_three_tiers`). On trn2.48xlarge without the UltraServer the second tier does not exist and traffic between instances goes over the EFA.
Mixture-of-experts models are sized without EP (experimental in Neuron 2.31) and the result says the expert part is uncalibrated.

### 7.5 Trn3 Gen1 and Gen2 (uncalibrated)

**Label on every Trn3 result: "uncalibrated: no published Trn3 training data".** The Trn3 pages give peak specifications only. Same assumptions as Trn2: Trn1's base
efficiency (pinned equal by a test) and NVLink's link efficiencies. **Unknown and kept unknown:** AWS instance type names (the calculator's labels `trn3-gen1-ultraserver`
and `trn3-gen2-ultraserver` are its own, and `trn3` alone is refused with the choice), the default LNC on Trn3 (the LNC page names LNC=2 as the default for Trainium2 only; LNC=2, 4 NeuronCores
per chip, is assumed and listed), price, usable HBM, and the efficiency of the switched tiers. Sources: [Trainium3](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium3.html),
[Trn3 UltraServers](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn3-arch.html).

| Field | Gen1 UltraServer | Gen2 UltraServer |
|---|---|---|
| Chips | 64 (4 servers of 16, 2 racks) | 144 (36 servers of 4, 2 racks of 18) |
| Per chip | 8 NeuronCore-v4, 671 BF16 TFLOPS dense, 144 GiB HBM at 4.9 TB/s; MXFP8 / MXFP4 2,517 TFLOPS (not used: MXFP8 training is experimental, Neuron 2.30 and 2.32, refused) | same |
| NeuronCores (LNC=2 assumed) | 4 per chip, 256 per UltraServer, 167.75 BF16 TFLOPS and 36 GiB each | 576, same |
| Tier 1 | paired chip, direct link, 1,024 GB/s per chip (2 chips) | server switch (radix 4), 1,024 GB/s per chip (4 chips) |
| Tier 2 | rack switch (radix 32), 1,024 GB/s per chip (32 chips) | rack switch (radix 36), 1,280 GB/s per chip (72 chips) |
| Tier 3 | same chip in the other rack, direct, 512 GB/s per chip (64 chips) | same chip in the other rack, direct, 256 GB/s per chip (144 chips) |
| NeuronLink total | 2,560 GB/s per chip (the three tiers add up to it) | 2,560 GB/s per chip |
| EFA | 12,800 Gbps per UltraServer = 25 GB/s per chip | 57,600 Gbps = 50 GB/s per chip |
| Billing | whole UltraServer, 64 chips | whole UltraServer, 144 chips |
| Price | unknown | unknown |

What changes against Trn2 (`mfu_model._tier_span_bw`): the tiers are switched, so a collective inside a tier gets the per-chip rate for any group shape (`test_switched_tiers_do_not_depend_on_the_group_shape`);
an expert-parallel all-to-all inside the 72-chip rack, and inside the whole UltraServer, uses no EFA, and only a group that spans two UltraServers does
(`test_ep_all_to_all_inside_the_rack_and_the_ultraserver_stays_off_the_efa`). The BF16 peak per chip (671 against 667 TFLOPS) is Trn2's, so the gain over Trn2 is memory (144 GiB, 4.9 TB/s)
and links, not BF16 compute; a Trn3 estimate has no data behind it.
