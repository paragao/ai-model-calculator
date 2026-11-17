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
"contact your AWS sales representative"). What is published is used. What is not stays unknown, is listed in the result
(`hardware.unknowns`) and is never filled in by a guess. The calculator's label is `p6e-gb300-ultraserver`, not an AWS
instance type: AWS has published no GB300 instance names.

| Field | Value | Source |
|---|---|---|
| Dense BF16 | 2,500 TFLOPS (derived: 360 PF with sparsity / 2 / 72). Same as GB200: the 1.5x uplift is FP4 only | [NVIDIA GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/), [AWS P6 page](https://aws.amazon.com/ec2/instance-types/p6/) ("1.5x ... FP4, without sparsity") |
| Dense FP8 | 5,000 TFLOPS (derived: 720 PF with sparsity / 2 / 72) | [NVIDIA GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/) |
| NVLink domain, bandwidth | 72 GPUs; 130 TB/s = 1.8 TB/s per GPU, 900 GB/s per direction | [NVIDIA GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/) |
| GPU memory | **279 GB per GPU is derived**: AWS says 1.5x the GPU memory of P6e-GB200 (13.4 TB), so 20.1 TB / 72. NVIDIA lists 20 TB per rack and 288 GB raw per GPU in partner datasheets. AWS gives no usable per-GPU figure | [AWS P6 page](https://aws.amazon.com/ec2/instance-types/p6/), [NVIDIA GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/) |
| SMs per GPU | 160 (the B300 die) | `mfu_model.SMS_PER_GPU` |
| EFA bandwidth per GPU | **Unknown.** `p6e-gb300` is not in the [AWS instance table](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html). NVIDIA's reference design has 800 Gb/s per GPU (ConnectX-8), which is not what AWS sells. The network entry is `None`: an estimate that needs traffic between UltraServers (more than 72 GPUs) is refused with that reason, one UltraServer is sized as usual, and the automatic GPU-count search stops at 72 and says so | none |
| Instance types and sizes | **Unknown.** The calculator assumes 4 GPUs as the smallest unit (the NVL72 compute tray holds 4) only for the GPU-count steps | [NVIDIA DGX SuperPOD GB300 reference architecture](https://www.hostcomponents.com/wp-content/uploads/2026/03/dgx-spod-gb300-ra.pdf) |
| On-demand price | **Unknown.** No cost line | none |

Validation: four GB300 / GB200 pairs of the NeMo 26.08.01 performance summary (`docs/CALCULATOR-MFU.md` section 7b). Ratio error
median 15.3%, worst 21.2%. The model predicts GB300 about equal to GB200 (same dense peak, one Blackwell efficiency), while the published
GB300 runs are 9-28% faster. A GB300 estimate is GB200-class and low by about that much. `dgx-gb300` (validation only) has 100 GB/s per GPU
(4 x ConnectX-8 800 Gb/s per 4-GPU compute tray).
