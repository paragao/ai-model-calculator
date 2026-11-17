"""
Expected-MFU model for LLM pre-training: architecture + parallel plan + hardware -> MFU,
throughput, and the parts that make it up.

ONE module, stdlib only, no globals read from elsewhere, so it can be ported 1:1
(docs/CALCULATOR-MFU.md lists every formula and constant with its source).

Conventions
-----------
- Parallel plan uses the Megatron-Core layout: N = TP x PP x CP x DP. EP is carved
  out of DP (EP divides DP x CP); expert weights are sharded EP ways (and TP ways).
- Model FLOPs per token (PaLM appendix B convention, fwd + bwd = 3 x fwd):
      F = 6 x N_matmul + 6 x H x S x (d_qk + d_v) x L_full + 12 x H_lin x d_lin^2 x L_linear
  N_matmul = active parameters that take part in a matmul (input embedding excluded,
  LM head included). Full attention is counted non-causal, as PaLM and Megatron do.
- MFU = F x tokens_per_step / (N x peak x step_time), peak = dense tensor peak of the
  training precision (BF16 or FP8).

Step-time model (per GPU, one optimizer step)
---------------------------------------------
  t_compute = F x tokens_per_step / N / (peak x eta)
      eta = 1 / ((1 - f_e) / eta_base + f_e / (eta_base x moe_factor))
      moe_factor = MOE_EXPERT_FACTOR x expert_ffn / (expert_ffn + MOE_EXPERT_HALF_FFN)
      f_e = share of F spent in routed-expert matmuls
  t_pipe    = t_compute / (1 - bubble),  bubble = (PP - 1) / (VP x m + PP - 1)
      m = micro-batches per DP rank per step = GBS / (DP x MBS)
  t_step    = t_pipe + exposed(TP) + exposed(CP) + exposed(EP) + exposed(DP)
      exposed(x) = max(t_x x (1 - OVERLAP_x), t_x - t_compute)
      t_x = bytes_x / (link_bandwidth x LINK_EFFICIENCY)
      ZeRO-3 / FSDP: exposed(DP) is summed per layer instead (_sharded_dp_time), because a
      layer's gather can only hide behind its neighbour's compute, not the whole step's.
Communication volumes are listed in each function below.
"""
from __future__ import annotations

import math

MODEL_VERSION = "mfu-model-4"

GIB = 1.073741824  # 1 GiB/s in GB/s (the Neuron pages give NeuronLink in GiB/s on Trn2)
CALIBRATION_TRN1 = ("calibrated on the 2 published Neuron Trn1 rows (Llama-3.1 8B and 70B, BF16, Neuron 2.24.0); there is no hold-out, "
                    "so no error on unseen runs is known")

CALIBRATION_TRN2 = ("uncalibrated: no published Trn2 training data (Trn1's base efficiency and NVLink's link efficiency are assumed, "
                    "so the throughput, days and cost are a model, not a forecast)")
UNKNOWNS_TRN2 = ("any measured Trn2 training throughput or MFU", "on-demand price", "usable HBM per NeuronCore after the Neuron runtime reserve",
                 "efficiency of the NeuronLink torus and ring")

CALIBRATION_TRN3 = ("uncalibrated: no published Trn3 training data (Trn1's base efficiency and NVLink's link efficiency are assumed, "
                    "so the throughput, days and cost are a model, not a forecast)")
UNKNOWNS_TRN3 = ("any measured Trn3 training throughput or MFU", "AWS instance type names", "on-demand price",
                 "default LNC on Trn3 (LNC=2, 4 NeuronCores per chip, is assumed as on Trn2)",
                 "usable HBM per NeuronCore after the Neuron runtime reserve", "efficiency of NeuronSwitch-v1 and the direct links")

# --- Hardware (dense tensor peak per GPU, TFLOPS; per-direction link bandwidth per GPU, GB/s) ---
# peak: NVIDIA datasheets (H100/H200/H800: 989 BF16, 1979 FP8 dense; A100: 312 BF16 dense;
#       B200 and B300: 2250 BF16, 4500 FP8 dense; HGX B300 = 36 PFLOPS FP8 dense / 8 GPUs).
# nvlink: per GPU total NVLink bandwidth / 2 (H100/H200 900 GB/s, B200/B300 1800 GB/s,
#       A100 600 GB/s, H800 400 GB/s).
# net: scale-out bandwidth per GPU = instance (node) network bandwidth / 8 GPUs.
#       AWS: p5/p5e/p5en/p6-b200 3200 Gbps EFA, p6-b300 6400 Gbps, p4d/p4de 400 Gbps.
#       DGX H100 / DGX B200: 8 x 400 Gb/s ConnectX-7; DGX B300: 8 x 800 Gb/s ConnectX-8; DGX A100: 8 x 200 Gb/s HDR;
#       Meta Llama 3 cluster: 400 Gbps RoCE per GPU; DeepSeek H800: 400 Gbps IB per GPU.
HARDWARE = {
    "p5.48xlarge":      {"gpu": "H100", "family": "hopper",    "bf16": 989.0,  "fp8": 1979.0, "nvlink": 450.0, "net": 50.0,  "gpn": 8},
    "p5e.48xlarge":     {"gpu": "H200", "family": "hopper",    "bf16": 989.0,  "fp8": 1979.0, "nvlink": 450.0, "net": 50.0,  "gpn": 8},
    "p5en.48xlarge":    {"gpu": "H200", "family": "hopper",    "bf16": 989.0,  "fp8": 1979.0, "nvlink": 450.0, "net": 50.0,  "gpn": 8},
    "p6-b200.48xlarge": {"gpu": "B200", "family": "blackwell", "bf16": 2250.0, "fp8": 4500.0, "nvlink": 900.0, "net": 50.0,  "gpn": 8},
    "p6-b300.48xlarge": {"gpu": "B300", "family": "blackwell", "bf16": 2250.0, "fp8": 4500.0, "nvlink": 900.0, "net": 100.0, "gpn": 8},
    "p4d.24xlarge":     {"gpu": "A100", "family": "ampere",    "bf16": 312.0,  "fp8": 312.0,  "nvlink": 300.0, "net": 6.25,  "gpn": 8},
    "p4de.24xlarge":    {"gpu": "A100", "family": "ampere",    "bf16": 312.0,  "fp8": 312.0,  "nvlink": 300.0, "net": 6.25,  "gpn": 8},
    # Non-AWS systems, used only by the validation set (published runs).
    "dgx-h100":         {"gpu": "H100", "family": "hopper",    "bf16": 989.0,  "fp8": 1979.0, "nvlink": 450.0, "net": 50.0,  "gpn": 8},
    "dgx-b200":         {"gpu": "B200", "family": "blackwell", "bf16": 2250.0, "fp8": 4500.0, "nvlink": 900.0, "net": 50.0,  "gpn": 8},
    "dgx-b300":         {"gpu": "B300", "family": "blackwell", "bf16": 2250.0, "fp8": 4500.0, "nvlink": 900.0, "net": 100.0, "gpn": 8},
    "dgx-a100-40g":     {"gpu": "A100", "family": "ampere",    "bf16": 312.0,  "fp8": 312.0,  "nvlink": 300.0, "net": 25.0,  "gpn": 8},
    "meta-h100-roce":   {"gpu": "H100", "family": "hopper",    "bf16": 989.0,  "fp8": 1979.0, "nvlink": 450.0, "net": 50.0,  "gpn": 8},
    "deepseek-h800":    {"gpu": "H800", "family": "hopper",    "bf16": 989.0,  "fp8": 1979.0, "nvlink": 200.0, "net": 50.0,  "gpn": 8},
    # --- Rack-scale systems: one NVLink domain of 72 (or 36) GPUs. ---------------------------------
    # Three sizes that are all "8" on an HGX node differ here, so each has its own field:
    #   domain_gpus   GPUs one NVLink domain reaches (what "gpn" means to the formulas below)
    #   instance_gpus GPUs of the smallest launchable instance (p6e-gb200.36xlarge: 4)
    #   billing_gpus  GPUs bought as one unit (a Capacity Block buys whole UltraServers)
    # GB200 (p6e-gb200), per GPU:
    #   bf16 2500, fp8 5000: dense; 360 PFLOPS FP8 dense / 72 GPUs
    #       https://aws.amazon.com/about-aws/whats-new/2025/07/amazon-p6e-gb200-ultraservers-gpu-performance-ec2
    #       (BF16 = half of FP8, as for B200; NVIDIA lists 360 PF sparse BF16 = 180 PF dense / 72 = 2500:
    #       https://www.nvidia.com/en-us/data-center/gb200-nvl72/)
    #   nvlink 900: 1.8 TB/s per GPU total NVLink5 / 2 (130 TB/s / 72 GPUs), same page
    #   net 100: p6e-gb200.36xlarge 3200 Gigabit EFA / 4 GPUs = 800 Gbps = 100 GB/s per GPU
    #       https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html (same column as every other row here)
    #   u-p6e-gb200x72 = 18 instances = 72 GPUs, u-p6e-gb200x36 = 9 instances = 36 GPUs (AWS launch page above)
    # GB300 (P6e-GB300 UltraServer), per GPU. Dense BF16 and FP8 equal GB200's: NVIDIA lists 360 PF BF16 and 720 PF FP8,
    # both with sparsity, for 72 GPUs (2500 / 5000 dense per GPU); the 1.5x uplift is FP4 only, as AWS says
    # ("1.5x ... (FP4, without sparsity)").
    #   https://www.nvidia.com/en-us/data-center/gb300-nvl72/  https://aws.amazon.com/ec2/instance-types/p6/
    #   nvlink 900: 130 TB/s / 72 GPUs = 1.8 TB/s per GPU total, same page. Domain 72.
    #   instance_gpus 4: the NVL72 compute tray holds 4 GPUs (NVIDIA DGX GB300 reference architecture); AWS has not
    #       published GB300 instance types or sizes.
    #   net 100: source "paragao, 2026-11-17" (topology given by the user, not published by AWS): one UltraServer = 72 B300 GPUs
    #       in one NVLink domain = 18 servers of 4 GPUs, each server with 8 EFA adapters at 400 Gbps = 3200 Gbps = 100 GB/s per GPU
    #       (the same per-GPU figure as GB200). EFA carries traffic only between UltraServers; inside the 72-GPU domain it is NVLink.
    # Left unknown on purpose (never filled in by a guess), listed in "unknowns":
    #   memory per GPU (279 GB is derived), price, instance type names and sizes: not published.
    # Board power (the "tgp_w" fields): NVIDIA lists the maximum board power (TGP) of Blackwell (B200, GB200) as up to 1,200 W
    # and of Blackwell Ultra (B300, GB300) as up to 1,400 W, at the same dense FP8 peak of 5 PFLOPS and the same 8 TB/s of HBM:
    #   https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/ (Table 2)
    # Large-GEMM transformer training draws the whole envelope (ViT-Large: 92-97% of TDP on B200 and B300, B300 +16% training,
    # +21% inference): arXiv 2610.41919. At equal listed peak, sustained tensor throughput follows the power headroom, see
    # TGP_EXPONENT. "tgp_ref_w" is the board power of the Blackwell part whose efficiency constant the system uses.
    # Applied to GB300 only. HGX B300 (1,100 W against 1,000 W) shows the same direction (+10% to +13% at the same plan in NeMo
    # 26.02 to 26.04.01), but the B300 rows of validation_runs.py are part of the fit of the Blackwell constants (BF16 and the MoE
    # factor), so the same term would push them 9 to 10 points up without a joint refit; see docs/CALCULATOR-HARDWARE.md section 6.
    "p6e-gb300-ultraserver": {"gpu": "GB300", "family": "blackwell", "bf16": 2500.0, "fp8": 5000.0, "nvlink": 900.0, "net": 100.0,
                              "gpn": 72, "domain_gpus": 72, "instance_gpus": 4, "billing_gpus": 72, "tgp_w": 1400.0, "tgp_ref_w": 1200.0,
                              "user_provided": ("EFA 100 GB/s per GPU, 18 servers of 4 GPUs with 8 x 400 Gbps EFA each (paragao, 2026-11-17)",),
                              "unknowns": ("usable GPU memory (279 GB is used: 1.5 x GB200's 13.4 TB / 72, AWS gives only the 1.5x)",
                                           "on-demand price", "instance type names and sizes")},
    # DGX GB300 NVL72 (validation only): 4 x ConnectX-8 800 Gb/s per 4-GPU compute tray = 100 GB/s per GPU
    #   https://www.nvidia.com/en-us/data-center/gb300-nvl72/ (800 Gb/s per GPU),
    #   NVIDIA DGX SuperPOD GB300 reference architecture (PDF), "The compute tray integrates four ConnectX-8 NICs ...
    #   QUANTUM-X800 (800Gbps)": https://www.hostcomponents.com/wp-content/uploads/2026/03/dgx-spod-gb300-ra.pdf
    "dgx-gb300":        {"gpu": "GB300", "family": "blackwell", "bf16": 2500.0, "fp8": 5000.0, "nvlink": 900.0, "net": 100.0,
                         "gpn": 72, "domain_gpus": 72, "instance_gpus": 4, "billing_gpus": 72, "tgp_w": 1400.0, "tgp_ref_w": 1200.0},
    # --- AWS Trainium (Trn1, Trn1n): one rank = one logical NeuronCore, as in the Neuron recipes (TP 32 = 32 cores = 16 chips) ---
    # The calculator counts chips ("GPUs" in the request and the result) and plans in ranks: TP, PP and DP degrees count NeuronCores.
    #   ranks_per_chip 2: Trainium has 2 NeuronCore-v2  https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/trainium.html
    #   bf16 95: 190 TFLOPS BF16 (dense) per chip / 2 cores, same page. fp8 None: FP8 training is refused (Neuron SDK 2.30 and 2.32:
    #       MXFP8 kernels in the NKI Library are experimental, https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/about-neuron/whats-new.html)
    #   memory 16 GiB per rank: 32 GiB per chip / 2 cores, same page (mem_gb is set in lambda_function._get_hardware_specs)
    #   16 chips per instance, NeuronLink-v2 2D torus, 384 GB/s per chip:
    #       https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/trn1-arch.html
    #       https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html (chip comparison table)
    #   nvlink 96: 384 GB/s per chip / 2 (per direction, as for NVLink) / 2 cores per chip. The tier is the whole instance: 32 ranks.
    #   net 3.125 (trn1) / 6.25 (trn1n): EFA 800 / 1600 Gbps per instance (https://aws.amazon.com/ec2/instance-types/trn1/) / 32 ranks, in GB/s.
    #   price: trn1.32xlarge 21.50 and trn1n.32xlarge 24.78 USD per hour on that AWS page (region not stated); see lambda_function.
    # Calibrated on the two Neuron Trn1 rows of validation_runs.TRN1_RUNS only: there is no hold-out (docs/CALCULATOR-HARDWARE.md section 7).
    "trn1.32xlarge":    {"gpu": "Trainium1", "family": "trainium1", "bf16": 95.0, "fp8": None, "nvlink": 96.0, "net": 3.125, "gpn": 32,
                         "ranks_per_chip": 2, "domain_gpus": 16, "instance_gpus": 16, "billing_gpus": 16,
                         "tiers": ((32, 96.0, "torus"),), "calibration": CALIBRATION_TRN1,
                         "unknowns": ("usable HBM per NeuronCore after the Neuron runtime reserve",)},
    "trn1n.32xlarge":   {"gpu": "Trainium1", "family": "trainium1", "bf16": 95.0, "fp8": None, "nvlink": 96.0, "net": 6.25, "gpn": 32,
                         "ranks_per_chip": 2, "domain_gpus": 16, "instance_gpus": 16, "billing_gpus": 16,
                         "tiers": ((32, 96.0, "torus"),), "calibration": CALIBRATION_TRN1,
                         "unknowns": ("usable HBM per NeuronCore after the Neuron runtime reserve",)},
    # --- Trainium2 (Trn2): uncalibrated, there is no published Trn2 training throughput or MFU (docs/CALCULATOR-HARDWARE.md section 7) ---
    # Sources: Trainium2 chip https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html
    #          Trn2 instances https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn2-arch.html
    #          LNC https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-features/logical-neuroncore-config.html
    #   ranks_per_chip 4: 8 physical NeuronCore-v3 per chip, LNC=2 (the default) groups them into 4 logical NeuronCores; 64 per trn2.48xlarge
    #   bf16 166.75: 667 BF16 TFLOPS per chip (dense) / 4. fp8 None: 1,299 cFP8 TFLOPS per chip exist, but FP8 training is refused (Neuron 2.30, 2.32)
    #   memory 24 GiB per rank: 96 GiB per chip / 4 (four 24 GB HBM banks, two physical cores each), set in lambda_function._get_hardware_specs
    #   16 chips per trn2.48xlarge / trn2u.48xlarge in a 4x4 2D torus; Trn2 UltraServer = 4 x trn2u.48xlarge = 64 chips, chips with the same
    #       coordinates in each instance form a ring across the four instances
    #   tier 1 (instance, 64 ranks): NeuronLink-v3 1,024 GiB/s per chip: 1024 x 1.0737 / 2 (per direction) / 4 ranks = 137.44 GB/s per rank
    #   tier 2 (UltraServer, 256 ranks): 256 GiB/s per chip between instances: 256 x 1.0737 / 2 / 4 = 34.36 GB/s per rank
    #   (the chip page's 1.28 TB/s per chip = 1,024 + 256 GiB/s, the same two tiers)
    #   net 6.25: EFAv3 3.2 Tbps per instance (16 x 200 Gigabit, AWS instance table) = 400 GB/s / 64 ranks; 25 GB/s per chip. The Neuron instance
    #       page prints "3,200 GB/s" in the same table as "3.2 Tbps": a unit slip for Gbps.
    #   price: no on-demand price on the AWS or Neuron pages checked: unknown, no cost line.
    # The base efficiency is Trn1's (BASE_EFFICIENCY trainium2 = trainium1: an assumption), the link efficiencies are NVLink's (assumption).
    "trn2.48xlarge":    {"gpu": "Trainium2", "family": "trainium2", "bf16": 166.75, "fp8": None, "nvlink": 137.44, "net": 6.25, "gpn": 64,
                         "ranks_per_chip": 4, "domain_gpus": 16, "instance_gpus": 16, "billing_gpus": 16,
                         "tiers": ((64, 137.44, "torus"),), "calibration": CALIBRATION_TRN2, "unknowns": UNKNOWNS_TRN2},
    "trn2-ultraserver": {"gpu": "Trainium2", "family": "trainium2", "bf16": 166.75, "fp8": None, "nvlink": 137.44, "net": 6.25, "gpn": 256,
                         "ranks_per_chip": 4, "domain_gpus": 64, "instance_gpus": 16, "billing_gpus": 64,
                         "tiers": ((64, 137.44, "torus"), (256, 34.36, "torus")), "calibration": CALIBRATION_TRN2,
                         "unknowns": UNKNOWNS_TRN2},
    # --- Trainium3 (Trn3 Gen1 and Gen2 UltraServers): uncalibrated, there is no published Trn3 training throughput or MFU ---
    # Sources: Trainium3 chip https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium3.html
    #          Trn3 UltraServers https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn3-arch.html
    #   chip: 8 NeuronCore-v4, 671 BF16 TFLOPS dense (42,944 / 64 chips on the Gen1 row, 96,624 / 144 on the Gen2 row), 2,517 MXFP8 / MXFP4 TFLOPS
    #       (FP8 and MXFP8 training are refused: experimental kernels, Neuron 2.30 and 2.32), 144 GiB HBM at 4.9 TB/s, NeuronLink-v4 2,560 GB/s per chip.
    #   ranks_per_chip 4: UNKNOWN default. The LNC page names LNC=2 (4 logical cores per chip) as the default for Trainium2 only; Trn3 supports
    #       LNC 1 or 2 and no page states its default. LNC=2 is assumed (listed in unknowns); it sets the NeuronCore count and the 36 GiB per rank.
    #   Switched, not a torus: NeuronSwitch-v1 connects chips all-to-all inside a tier, one memory space per UltraServer (Trn3 architecture page).
    #   Gen1 UltraServer, 64 chips = 4 servers of 16 chips, 2 racks: tier 1 the paired chip, direct NeuronLink-v4, 1,024 GB/s per chip (2 chips = 8 ranks);
    #       tier 2 the rack switch (radix 32), 1,024 GB/s per chip (32 chips = 128 ranks); tier 3 the same chip in the other rack, direct link,
    #       512 GB/s per chip (64 chips = 256 ranks). Per rank per direction: 1,024 / 2 / 4 = 128, 512 / 2 / 4 = 64 GB/s.
    #   Gen2 UltraServer, 144 chips = 36 servers of 4 chips, 2 racks of 18 servers: tier 1 the server switch (radix 4), 1,024 GB/s per chip (4 chips = 16 ranks);
    #       tier 2 the rack switch (radix 36), 1,280 GB/s per chip (72 chips = 288 ranks); tier 3 the same chip in the other rack, direct link,
    #       256 GB/s per chip (144 chips = 576 ranks). Per rank per direction: 128, 160 and 32 GB/s.
    #   net (EFA per UltraServer): Gen1 12,800 Gbps / 64 chips = 200 Gbps = 25 GB/s per chip = 6.25 GB/s per rank; Gen2 57,600 / 144 = 400 Gbps = 50 GB/s
    #       per chip = 12.5 per rank. AWS has published no Trn3 instance type names: the keys are the calculator's own labels.
    #   price: unknown, no cost line. Efficiency: Trn1's (assumption), the link efficiencies are NVLink's (assumption).
    "trn3-gen1-ultraserver": {"gpu": "Trainium3", "family": "trainium3", "bf16": 167.75, "fp8": None, "nvlink": 128.0, "net": 6.25, "gpn": 256,
                              "ranks_per_chip": 4, "domain_gpus": 64, "instance_gpus": 16, "billing_gpus": 64,
                              "tiers": ((8, 128.0, "direct"), (128, 128.0, "switch"), (256, 64.0, "direct")),
                              "calibration": CALIBRATION_TRN3, "unknowns": UNKNOWNS_TRN3},
    "trn3-gen2-ultraserver": {"gpu": "Trainium3", "family": "trainium3", "bf16": 167.75, "fp8": None, "nvlink": 160.0, "net": 12.5, "gpn": 576,
                              "ranks_per_chip": 4, "domain_gpus": 144, "instance_gpus": 4, "billing_gpus": 144,
                              "tiers": ((16, 128.0, "switch"), (288, 160.0, "switch"), (576, 32.0, "direct")),
                              "calibration": CALIBRATION_TRN3, "unknowns": UNKNOWNS_TRN3},
    # DGX GB200 NVL72 (validation only): 18 compute trays x 4 GPUs, each tray 4 x ConnectX-7 NDR 400 Gb/s = 50 GB/s per GPU
    #   https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/dgx-superpod-components.html
    "dgx-gb200":        {"gpu": "GB200", "family": "blackwell", "bf16": 2500.0, "fp8": 5000.0, "nvlink": 900.0, "net": 50.0,
                         "gpn": 72, "domain_gpus": 72, "instance_gpus": 4, "billing_gpus": 72},
    "u-p6e-gb200x72":   {"gpu": "GB200", "family": "blackwell", "bf16": 2500.0, "fp8": 5000.0, "nvlink": 900.0, "net": 100.0,
                         "gpn": 72, "domain_gpus": 72, "instance_gpus": 4, "billing_gpus": 72},
    "u-p6e-gb200x36":   {"gpu": "GB200", "family": "blackwell", "bf16": 2500.0, "fp8": 5000.0, "nvlink": 900.0, "net": 100.0,
                         "gpn": 36, "domain_gpus": 36, "instance_gpus": 4, "billing_gpus": 36},
}


def domain_gpus(hw):
    """GPUs one NVLink domain reaches: 8 on an HGX node, 72 on a GB200 NVL72 UltraServer."""
    return int(hw.get("domain_gpus") or hw["gpn"])


def instance_gpus(hw):
    """GPUs of the smallest launchable instance (4 for p6e-gb200.36xlarge, else the node)."""
    return int(hw.get("instance_gpus") or hw["gpn"])


def billing_gpus(hw):
    """GPUs bought as one unit: the node, or the whole UltraServer."""
    return int(hw.get("billing_gpus") or hw["gpn"])


# --- Rack-aware GPU counts (hardware whose NVLink domain is larger than 8 GPUs) -----------------
# 72 is not a power of two. A job that wants EP 64 (or any power-of-two layout) on an UltraServer uses 64 of
# its 72 GPUs, as NeMo's GB200 recipes do (64, 128 or 256 GPUs; NeMo 25.07 performance summary), and is still
# billed for the whole UltraServer. HGX nodes (domain 8) are never touched by these helpers.

def is_trainium(hw):
    """Hardware counted in chips with NeuronCores as ranks (hw["ranks_per_chip"])."""
    return bool(hw and hw.get("ranks_per_chip"))


def ranks_per_chip(hw):
    return int(hw.get("ranks_per_chip") or 1)


def is_rack_scale(hw):
    return domain_gpus(hw) > 8 and not is_trainium(hw)


def _largest_pow2_below(n, multiple_of):
    p = 1
    while p * 2 < n:
        p *= 2
    while p >= multiple_of and p % multiple_of:
        p //= 2
    return p if p >= multiple_of else 0


def rack_share(hw):
    """GPUs of each domain a power-of-two layout uses: 64 of 72 (0 for HGX nodes)."""
    return _largest_pow2_below(domain_gpus(hw) + 1, instance_gpus(hw)) if is_rack_scale(hw) else 0


def rack_hw(hw, n):
    """`hw` as seen by a job of n GPUs. On rack-scale hardware, n GPUs that do not fill whole domains (128 on
    72-GPU UltraServers) run as the largest power-of-two share of each domain that divides n (64 per UltraServer),
    so a group of that size still sits on NVLink. Any other case, and every HGX node, returns `hw` itself."""
    if not is_rack_scale(hw):
        return hw
    domain = domain_gpus(hw)
    if n <= domain or n % domain == 0:
        return hw
    share = rack_share(hw)
    if share and n % share == 0:
        return {**hw, "gpn": share}
    return hw


def rack_alternatives(hw, n):
    """GPU counts below n worth trying instead of n on rack-scale hardware: the largest power of two (a multiple of
    the instance size) when n is not one, e.g. 64 for 72 and 128 for 144. Empty for HGX nodes."""
    if not is_rack_scale(hw) or n <= instance_gpus(hw):
        return []
    p = _largest_pow2_below(n, instance_gpus(hw))
    return [p] if p and p != n and (n & (n - 1)) else []


def auto_gpu_counts(hw, gpus_per_node, limit):
    """GPU counts the automatic search tries, smallest first. HGX nodes: gpus_per_node, doubling. Rack-scale:
    steps of the instance size inside one domain (4, 8, 16, 32, 64), then whole domains (72, 144, 288, ...) and
    the power-of-two share of each (128, 256, ...)."""
    if is_trainium(hw):  # chips: doubling inside one domain, then whole domains (Trn2 UltraServer 64, Trn3 144 chips)
        step, domain = instance_gpus(hw), domain_gpus(hw)
        out, n = [], step
        while n <= min(domain, limit):
            out.append(n)
            n *= 2
        n = domain
        while n <= limit:
            if n not in out:
                out.append(n)
            n *= 2
        return sorted(out)
    if not is_rack_scale(hw):
        out, n = [], gpus_per_node
        while n <= limit:
            out.append(n)
            n *= 2
        return out
    domain, step = domain_gpus(hw), instance_gpus(hw)
    share = rack_share(hw)
    counts = set()
    for base in (step, share, domain):
        n = base
        while n <= limit:
            if base == step and n > domain:
                break
            counts.add(n)
            n *= 2
    return sorted(counts)

# --- Calibration (fitted by validation_runs.fit() on the FIT subset; see docs) ---------------
# eta_base: fraction of dense peak reached by the per-GPU compute of the step, before bubble
# and exposed communication. One value per GPU family and precision, never per model.
BASE_EFFICIENCY = {
    ("hopper", "bf16"): 0.573,
    ("hopper", "fp8"): 0.45,
    ("blackwell", "bf16"): 0.443,
    ("blackwell", "fp8"): 0.417,
    ("ampere", "bf16"): 0.471,
    ("trainium1", "bf16"): 0.33,  # fitted on the two Neuron Trn1 rows (validation_runs.fit_trainium1)
    ("trainium2", "bf16"): 0.33,  # NOT fitted: Trn1's value, an assumption (no published Trn2 training data); test pins it equal
    ("trainium3", "bf16"): 0.33,  # NOT fitted: Trn1's value, an assumption (no published Trn3 training data); test pins it equal
}
# Routed-expert matmuls (grouped GEMMs, permute/unpermute, routing) run at
#   moe_factor = MOE_EXPERT_FACTOR x expert_ffn / (expert_ffn + MOE_EXPERT_HALF_FFN)
# of eta_base: small experts (narrow GEMMs) are less efficient than wide ones.
MOE_EXPERT_FACTOR = 0.78
MOE_EXPERT_HALF_FFN = 1024
# Sustained tensor throughput at equal listed peak scales with board power: for a system with "tgp_w" and "tgp_ref_w",
#   eta_base = BASE_EFFICIENCY[family, precision] x (tgp_w / tgp_ref_w) ** TGP_EXPONENT
# One constant, fitted on the two dense FP8 GB300 / GB200 pairs of NeMo 26.04.01 (validation_runs.fit_tgp_exponent), checked on the
# other GB300 pairs. An exponent above 1 means the power ratio explains most of the gap and a small part (about 3 to 4 points
# for GB300, 1,400 W against 1,200 W) comes from elsewhere, see docs/CALCULATOR-HARDWARE.md section 6.
TGP_EXPONENT = 1.25

# --- Stated assumptions (not fitted) ---------------------------------------------------------
LINK_EFFICIENCY = {"nvlink": 0.76, "net": 0.80, "torus": 0.76, "switch": 0.76, "direct": 0.76}
# torus, switch, direct (NeuronLink, NeuronSwitch, direct chip-to-chip links): assumption, no Neuron measurement exists; set equal to nvlink. Only the base efficiency of the Trainium
#   families is fitted (validation_runs.fit_trainium1), not the link efficiencies.
# nvlink 0.76: our DeepEP intranode dispatch 340 GB/s on p5en vs 450 GB/s per direction
#   (benchmarks/ep-dispatch-combine-bf16/p5en-2nodes). net 0.80: assumption (NCCL ring busbw).
OVERLAP = {"tp": 0.5, "cp": 0.9, "ep": 0.3, "dp": 0.9, "fsdp": 0.8}
# Share of each communication hidden behind compute (assumptions: Megatron-Core TP-comm
# overlap, ring-attention CP, distributed-optimizer grad/param overlap). EP all-to-all is not
# overlapped by default (Megatron-Core 1F1B); a plan may set overlap_ep (DualPipe: 0.95).
GRAD_BYTES = 2      # BF16 gradient reduce-scatter
PARAM_BYTES = 2     # BF16 parameter all-gather
ACT_BYTES = 2       # BF16 activations on the wire (FP8 dispatch not modelled)

# Wall-clock goodput: share of wall time spent in training steps (restarts, checkpoints and
# evaluation take the rest). Llama 3 paper, Sec. 3.3.4: "higher than 90% effective training time".
GOODPUT = 0.90

# Streaming multiprocessors per GPU (for the SMs a communication kernel takes from compute).
# H100/H200/H800: 132 (NVIDIA H100 Tensor Core GPU Architecture whitepaper, H100 SXM5);
# A100: 108 (NVIDIA A100 whitepaper); B200: 148 (74 per die, chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut);
# B300: 160 (NVIDIA "Inside NVIDIA Blackwell Ultra" blog: up to 160, varies by SKU).
# GB200 carries the B200 die (148 SMs, same source); the Blackwell Ultra die of GB300 is the B300 die (160).
SMS_PER_GPU = {"H100": 132, "H200": 132, "H800": 132, "A100": 108, "B200": 148, "B300": 160, "GB200": 148, "GB300": 160}
# Token dispatchers whose all-to-all runs as SM kernels (DeepEP, NVIDIA HybridEP). When the plan
# overlaps the all-to-all with compute (plan a2a_overlap), those SMs are not available to the GEMMs.
SM_DISPATCHERS = ("deepep", "hybridep")
COMM_SMS = 20  # DeepSeek-V3 paper Sec. 3.2.2: 20 of the H800's 132 SMs run the all-to-all (plan dispatcher_sms overrides)
# Activation recompute (plan recompute = Megatron-Core recompute_modules): each listed module's forward
# runs again in backward. Elementwise modules (layernorm, moe_act) add no matmul FLOPs.
RECOMPUTE_MODULES = ("mlp", "moe", "shared_experts", "mla_up_proj", "core_attn", "layernorm", "moe_act")

# Default global batch when the request gives none: 16M tokens per step, the Llama 3 405B
# setting (Llama 3 paper, Table 4); at least one micro-batch per DP rank.
DEFAULT_TOKENS_PER_STEP = 16 * 1024 * 1024

GATED_ACTIVATIONS = ("silu", "swish", "swiglu", "geglu", "gelu_pytorch_tanh")

# Pinned result of validation_runs.py (test_mfu_model checks it is current).
# Scored runs only: superseded runs are printed but not scored (validation_runs.summary).
VALIDATION_SUMMARY = {"runs": 38, "hold_out_runs": 23, "median_abs_error_pct": 6.1, "max_abs_error_pct": 58.9,
                      "hold_out_median_abs_error_pct": 7.4, "hold_out_max_abs_error_pct": 58.9,
                      "original_runs": 27, "original_median_abs_error_pct": 5.7, "original_max_abs_error_pct": 19.8,
                      "new_moe_runs": 11, "new_moe_median_abs_error_pct": 12.2, "new_moe_max_abs_error_pct": 58.9,
                      "superseded_runs": 2,
                      "trainium1_rows": 2, "trainium1_median_abs_error_pct": 13.1, "trainium1_max_abs_error_pct": 13.5,
                      "rack_scale_runs": 8, "rack_scale_median_abs_error_pct": 10.4, "rack_scale_max_abs_error_pct": 82.3,
                      "gb300_fit_pairs": 2, "gb300_pairs": 10, "gb300_pair_median_abs_error_pct": 3.8, "gb300_pair_max_abs_error_pct": 7.1,
                      "metric": "tokens/s per GPU or MFU vs published", "doc": "docs/CALCULATOR-MFU.md"}

SOURCES = {
    "llama3_table4": "https://arxiv.org/abs/2407.21783 (The Llama 3 Herd of Models, Table 4)",
    "palm_flops": "https://arxiv.org/abs/2204.02311 (PaLM, Appendix B: model FLOPs)",
    "megatron_bubble": "https://arxiv.org/abs/2104.04473 (Narayanan et al. 2021: bubble (p-1)/(v m))",
    "nemo_perf": "https://docs.nvidia.com/nemo-framework/user-guide/25.11/performance/archive/25.07-performance-summary.html",
}


# === Shape: any Hugging Face config.json -> parameter and FLOP counts ==========================

def _num(cfg, *keys, default=0):
    for k in keys:
        v = cfg.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
            return int(v)
    return default


def ffn_matrices(cfg):
    """3 for a gated FFN (SwiGLU/GeGLU), 2 for a plain one (GELU/ReLU/ReLU^2).

    GPT-2/OPT/BLOOM-style configs name the activation `activation_function`; those are gated
    only if the name contains 'glu'. Otherwise hidden_act / hidden_activation decides; an
    unknown or missing activation counts as gated (Llama-family default).
    """
    if isinstance(cfg.get("activation_function"), str):
        return 3 if "glu" in cfg["activation_function"].lower() else 2
    act = str(cfg.get("hidden_act") or cfg.get("hidden_activation") or "").lower()
    if not act:
        return 3
    return 3 if act in GATED_ACTIVATIONS or "glu" in act else 2


def moe_layer_count(cfg, layers, n_experts):
    """MoE layers: explicit count, every k-th layer, or all after the first dense ones."""
    if not n_experts:
        return 0
    if _num(cfg, "moe_layers"):
        return min(layers, _num(cfg, "moe_layers"))
    step = _num(cfg, "decoder_sparse_step", "interleave_moe_layer_step", "moe_layer_freq")
    if step > 1:
        return layers // step
    return layers - min(layers, _num(cfg, "first_k_dense_replace"))


def attention_shape(cfg, d, layers):
    """Attention parameters (all layers) and the per-layer dims that set its FLOPs."""
    heads = _num(cfg, "num_attention_heads", "n_head", default=max(1, d // 128))
    kv_heads = _num(cfg, "num_key_value_heads", "num_kv_heads", default=heads)
    if cfg.get("multi_query") is True:
        kv_heads = 1
    head_dim = _num(cfg, "head_dim", default=d // heads)
    kv_lora = _num(cfg, "kv_lora_rank")
    if kv_lora:  # MLA
        rope = _num(cfg, "qk_rope_head_dim")
        qk = (_num(cfg, "qk_nope_head_dim") + rope) or _num(cfg, "qk_head_dim") or head_dim
        v = _num(cfg, "v_head_dim", default=qk)
        q_lora = _num(cfg, "q_lora_rank")
        q = d * q_lora + q_lora * heads * qk if q_lora else d * heads * qk
        full = q + d * (kv_lora + rope) + kv_lora * heads * (qk - rope + v) + heads * v * d
        mla_up = (q_lora * heads * qk if q_lora else 0) + kv_lora * heads * (qk - rope + v)
        kind = "MLA"
    else:
        qk = v = head_dim
        mla_up = 0
        full = d * head_dim * (2 * heads + 2 * kv_heads)
        kind = "MQA" if kv_heads == 1 and heads > 1 else ("GQA" if kv_heads < heads else "MHA")
    linear = min(layers, _num(cfg, "linear_attn_layers"))
    lin_heads = _num(cfg, "linear_attn_heads", default=heads)
    lin_dim = _num(cfg, "linear_attn_head_dim", default=head_dim)
    params = full * (layers - linear) + 4 * d * lin_heads * lin_dim * linear
    if linear:
        kind = f"{kind} in {layers - linear} layers, linear attention in {linear}"
    return {"kind": kind, "params": params, "per_layer_avg": params // layers, "heads": heads,
            "mla_up_per_layer": mla_up,
            "kv_heads": kv_heads, "d_qk": qk, "d_v": v, "full_layers": layers - linear,
            "linear_layers": linear, "lin_heads": lin_heads, "lin_dim": lin_dim}


def model_shape(cfg, params_total_b=None):
    """Parameter counts (total, active, experts) and FLOP inputs of any decoder-only config.

    params_total_b (an exact count, e.g. the safetensors index) rescales the totals; the shape
    sum is kept as shape_total_b. Returns plain numbers only.
    """
    layers = _num(cfg, "num_hidden_layers", "num_layers", "n_layer")
    d = _num(cfg, "hidden_size", "d_model", "n_embd")
    if not layers or not d:
        raise ValueError("architecture needs num_hidden_layers and hidden_size")
    vocab = _num(cfg, "vocab_size")
    tied = cfg.get("tie_word_embeddings", True) is not False
    mats = ffn_matrices(cfg)
    dense_ffn = _num(cfg, "intermediate_size", "ffn_dim", "ffn_hidden_size", default=4 * d)
    n_experts = _num(cfg, "n_routed_experts", "num_experts", "num_local_experts")
    topk = _num(cfg, "num_experts_per_tok", "moe_topk", "top_k", default=1) if n_experts else 0
    expert_ffn = _num(cfg, "moe_intermediate_size", default=dense_ffn) if n_experts else 0
    shared_ffn = 0
    if n_experts:
        shared_ffn = (_num(cfg, "shared_expert_intermediate_size")
                      or _num(cfg, "n_shared_experts", "num_shared_experts") * expert_ffn
                      or _num(cfg, "moe_shared_expert_intermediate_size"))
    moe_layers = moe_layer_count(cfg, layers, n_experts)
    dense_layers = layers - moe_layers
    attn = attention_shape(cfg, d, layers)

    expert_each = mats * d * expert_ffn
    shared = mats * d * shared_ffn
    router = d * n_experts
    dense_mlp = mats * d * dense_ffn
    embed_in = vocab * d
    lm_head = 0 if tied else vocab * d
    expert_total = moe_layers * n_experts * expert_each
    nonexpert = attn["params"] + dense_layers * dense_mlp + moe_layers * (shared + router) + embed_in + lm_head
    shape_total = nonexpert + expert_total
    expert_active = moe_layers * topk * expert_each
    active = nonexpert + expert_active
    scale = (float(params_total_b) * 1e9 / shape_total) if params_total_b else 1.0
    # Matmul parameters per token: everything active except the input-embedding lookup
    # (a tied LM head is still a matmul over the vocabulary).
    matmul = active - embed_in + (embed_in if tied else 0)
    # Multi-token prediction (num_nextn_predict_layers, DeepSeek-V3 / GLM): each MTP module is one
    # more decoder layer, a 2d -> d projection and an LM-head pass. Compute only: checkpoints
    # count these weights separately from the published total.
    mtp = _num(cfg, "num_nextn_predict_layers")
    layer_active = attn["per_layer_avg"] + ((shared + router + topk * expert_each) if n_experts else dense_mlp)
    mtp_matmul = mtp * (layer_active + 2 * d * d + vocab * d)
    mtp_expert = mtp * topk * expert_each if n_experts else 0
    return {
        "layers": layers, "d": d, "vocab": vocab, "tied_embeddings": tied, "ffn_matrices": mats,
        "dense_ffn": dense_ffn, "dense_layers": dense_layers, "moe_layers": moe_layers,
        "n_experts": n_experts, "topk": topk,
        "n_group": _num(cfg, "n_group", default=1), "topk_group": _num(cfg, "topk_group", default=topk), "expert_ffn": expert_ffn, "shared_ffn": shared_ffn,
        "attention": attn,
        "attn_per_layer": attn["per_layer_avg"], "expert_each": expert_each, "shared_each": shared,
        "router_each": router, "dense_layer_params": attn["per_layer_avg"] + dense_mlp,
        "moe_layer_params": n_experts * expert_each + shared + router,
        "shape_total_b": round(shape_total / 1e9, 2),
        "total_b": round(shape_total * scale / 1e9, 2),
        "active_b": round(active * scale / 1e9, 2),
        "expert_total_b": round(expert_total * scale / 1e9, 4),
        "matmul_active": (matmul + mtp_matmul) * scale,
        "expert_active_matmul": (expert_active + mtp_expert) * scale,
        "mtp_layers": mtp,
        "scale": scale,
    }


def moe_expert_factor(expert_ffn):
    return MOE_EXPERT_FACTOR * expert_ffn / (expert_ffn + MOE_EXPERT_HALF_FFN)


def flops_per_token(shape, seq):
    """Training model FLOPs per token (fwd + bwd), PaLM convention (see module docstring)."""
    a = shape["attention"]
    dense = 6.0 * shape["matmul_active"]
    attn = 6.0 * a["heads"] * seq * (a["d_qk"] + a["d_v"]) * a["full_layers"]
    linear = 12.0 * a["lin_heads"] * a["lin_dim"] ** 2 * a["linear_layers"]
    return {"total": dense + attn + linear, "matmul": dense, "attention": attn + linear,
            "experts": 6.0 * shape["expert_active_matmul"]}


# === Plan -> step time and MFU ==================================================================

def _bw(hw, kind):
    return hw[kind] * LINK_EFFICIENCY[kind] * 1e9  # bytes/s


def _tier_span_bw(hw, inner, group):
    """Collective bandwidth when the hardware has link tiers (Trainium): the group spans inner x group ranks.
    tiers: ((ranks reached, GB/s per rank per direction, kind), ...) from the innermost out. The smallest tier that holds the span
    carries it. Past the last tier the traffic leaves through the EFA: the group has r = last tier size / inner ranks in each
    domain and each brings its own adapter share (min of the last tier and r x EFA per rank), as for the GPUs."""
    span = inner * group
    for size, bw, kind in hw["tiers"]:
        if span <= size:
            return bw * LINK_EFFICIENCY[kind] * 1e9
    size, bw, kind = hw["tiers"][-1]
    r = max(1, size // inner)
    return min(bw * LINK_EFFICIENCY[kind], r * hw["net"] * LINK_EFFICIENCY["net"]) * 1e9


def _span_bw(hw, inner, group):
    """Collective bandwidth of `group` ranks spaced `inner` apart (ring/tree, NCCL-style).

    Inside one node: NVLink. Across nodes: the group has r = gpn / inner ranks on each node and
    its traffic leaves through those r GPUs' NICs, so bandwidth = min(NVLink, r x NIC per GPU).
    """
    if hw.get("tiers"):
        return _tier_span_bw(hw, inner, group)
    if inner * group <= hw["gpn"]:
        return _bw(hw, "nvlink")
    r = max(1, hw["gpn"] // inner)
    return min(hw["nvlink"] * LINK_EFFICIENCY["nvlink"], r * hw["net"] * LINK_EFFICIENCY["net"]) * 1e9


def default_global_batch(seq, dp, mbs, tokens_per_step=DEFAULT_TOKENS_PER_STEP):
    """Sequences per step: tokens_per_step / seq, rounded to whole micro-batches per DP rank (>= 1)."""
    per_rank = max(1, round(tokens_per_step / seq / (dp * mbs)))
    return per_rank * dp * mbs


def recompute_flops_per_token(shape, seq, modules):
    """Forward FLOPs per token run again by activation recompute of `modules` (2 x the matmul params of
    each module, or the forward third of the attention-score FLOPs for core_attn)."""
    unknown = [x for x in modules if x not in RECOMPUTE_MODULES]
    if unknown:
        raise ValueError(f"unknown recompute module(s) {unknown}; known: {', '.join(RECOMPUTE_MODULES)}")
    a, sc = shape["attention"], shape.get("scale", 1.0)
    f = 0.0
    if "mlp" in modules:  # dense-layer MLPs
        f += 2.0 * (shape.get("dense_layer_params", 0) - shape.get("attn_per_layer", 0)) * shape.get("dense_layers", 0) * sc
    if "moe" in modules:  # whole MoE block: active routed experts, shared expert, router
        f += 2.0 * shape["moe_layers"] * (shape["topk"] * shape.get("expert_each", 0) + shape.get("shared_each", 0)
                                          + shape.get("router_each", 0)) * sc
    elif "shared_experts" in modules:
        f += 2.0 * shape["moe_layers"] * shape.get("shared_each", 0) * sc
    if "mla_up_proj" in modules:
        f += 2.0 * a.get("mla_up_per_layer", 0) * a["full_layers"] * sc
    if "core_attn" in modules:
        f += 2.0 * a["heads"] * seq * (a["d_qk"] + a["d_v"]) * a["full_layers"]
    return f


def _sharded_dp_time(shape, hw, seq, t_compute, m, tp, pp, etp, ep, g_dense, g_expert, prefetch):
    """ZeRO-3 / FSDP data-parallel time per step: (total, exposed).

    Units: a dense layer, an MoE layer, and the embedding + LM head. For each unit u, each micro-batch:
        AG_u = (g-1)/g x params_u x PARAM_BYTES / bw(g),  RS_u = the same with GRAD_BYTES
        (dense params over the DP x CP group, expert params over the expert-DP group).
    A gather can only hide behind the compute of the neighbouring unit in the same pass, about 1/3
    (forward) and 2/3 (backward) of the unit's training compute c_u, and only as far as the prefetch
    buffer allows (prefetch 0..1, from the memory model's ZeRO-3 live-parameter cap):
        exposed_fwd = max(AG x (1 - OVERLAP_fsdp), AG - prefetch x c_u / 3)
        exposed_bwd = max((AG + RS) x (1 - OVERLAP_fsdp), AG + RS - prefetch x 2 c_u / 3)
    """
    a = shape["attention"]
    layers, d, sc = shape["layers"], shape["d"], shape["scale"]
    # Attention-score FLOPs per token, spread evenly over the layers (PaLM convention, as flops_per_token).
    attn_layer = (6.0 * a["heads"] * seq * (a["d_qk"] + a["d_v"]) * a["full_layers"]
                  + 12.0 * a["lin_heads"] * a["lin_dim"] ** 2 * a["linear_layers"]) / layers
    units = []  # (units on one stage, dense-group params per rank, expert-group params per rank, FLOPs/token)
    if shape["dense_layers"]:
        p = shape["dense_layer_params"] * sc
        units.append((shape["dense_layers"] / pp, p / tp, 0.0, 6.0 * p + attn_layer))
    if shape["moe_layers"]:
        nonexp = (shape["attn_per_layer"] + shape["shared_each"] + shape["router_each"]) * sc
        active = nonexp + shape["topk"] * shape["expert_each"] * sc
        units.append((shape["moe_layers"] / pp, nonexp / tp,
                      shape["n_experts"] * shape["expert_each"] * sc / (etp * max(ep, 1)), 6.0 * active + attn_layer))
    head = shape["vocab"] * d * sc
    units.append((1.0 / pp, head * (1 if shape["tied_embeddings"] else 2) / tp, 0.0, 6.0 * head))
    fl_total = sum(c * f for c, _, _, f in units) or 1.0

    def gather(params, g):
        return (g - 1) / g * params / _span_bw(hw, tp, g) if g > 1 and params else 0.0

    total = exposed_sum = 0.0
    ov = OVERLAP["fsdp"]
    for count, p_dense, p_expert, f in units:
        base = gather(p_dense, g_dense) + gather(p_expert, g_expert)
        ag, rs = base * PARAM_BYTES, base * GRAD_BYTES
        c_u = t_compute / m * f / fl_total  # this unit's compute per micro-batch
        e_fwd = max(ag * (1 - ov), ag - prefetch * c_u / 3)
        e_bwd = max((ag + rs) * (1 - ov), ag + rs - prefetch * 2 * c_u / 3)
        total += count * m * (2 * ag + rs)
        exposed_sum += count * m * (e_fwd + e_bwd)
    return total, exposed_sum


def estimate(shape, hw, plan, precision="bf16"):
    """Expected step time, throughput and MFU of one plan, with every part.

    plan: num_gpus, tp, pp, cp, vp, ep, mbs, seq, gbs (sequences; None = default),
          zero (1 = distributed optimizer, 2 = grads reduce-scattered per micro-batch,
                3 = params sharded too: gathered for forward and backward every micro-batch),
          fsdp (bool: same as zero 3),
          zero3_prefetch (0..1, default 1: share of the next unit's gather that can run ahead,
                from the memory model's ZeRO-3 live-parameter cap),
          recompute (Megatron-Core recompute_modules: forward FLOPs added to hardware time),
          dispatcher, a2a_overlap, dispatcher_sms (an overlapped DeepEP / HybridEP all-to-all takes
                dispatcher_sms, default COMM_SMS, of the GPU's SMs from compute).
    """
    n = int(plan["num_gpus"])
    hw = rack_hw(hw, n)
    tp, pp, cp = int(plan.get("tp", 1)), int(plan.get("pp", 1)), int(plan.get("cp", 1))
    vp, ep, mbs = int(plan.get("vp", 1)), int(plan.get("ep", 1)), int(plan.get("mbs", 1))
    seq = int(plan["seq"])
    if n % (tp * pp * cp):
        raise ValueError(f"TP x PP x CP = {tp * pp * cp} does not divide {n} GPUs")
    dp = n // (tp * pp * cp)
    # Expert layout (Megatron-Core): N / PP = ETP x EP x expert-DP. Expert TP = TP when that
    # divides, else 1 (as the NeMo DeepSeek-V3 / Mixtral recipes set it).
    per_stage = n // pp
    etp = int(plan.get("etp") or (tp if per_stage % (tp * ep) == 0 else 1))
    if per_stage % (etp * ep):
        raise ValueError(f"EP {ep} x expert TP {etp} does not divide {per_stage} GPUs per pipeline stage")
    prec = "fp8" if str(precision).lower() == "fp8" else "bf16"
    if hw[prec] is None:
        raise ValueError(f"{prec.upper()} training on {hw['gpu']} is not sized: FP8 / MXFP8 training kernels are experimental in the Neuron SDK "
                         "(2.30 and 2.32). Use BF16.")
    peak = hw[prec] * 1e12
    gbs_given = plan.get("gbs")
    gbs = int(gbs_given) if gbs_given else default_global_batch(seq, dp, mbs)
    m = max(1.0, gbs / (dp * mbs))
    tokens_step = gbs * seq

    fl = flops_per_token(shape, seq)
    f_gpu = fl["total"] * tokens_step / n
    eta_base = BASE_EFFICIENCY.get((hw["family"], prec)) or BASE_EFFICIENCY[(hw["family"], "bf16")]
    tgp_gain = (hw["tgp_w"] / hw["tgp_ref_w"]) ** TGP_EXPONENT if hw.get("tgp_ref_w") else 1.0
    eta_base = eta_base * tgp_gain
    f_e = fl["experts"] / fl["total"] if fl["total"] else 0.0
    moe_factor = moe_expert_factor(shape["expert_ffn"]) if f_e else 1.0
    eta = 1.0 / ((1 - f_e) / eta_base + f_e / (eta_base * moe_factor))
    t_compute = f_gpu / (peak * eta)
    # Recompute: hardware runs more FLOPs than the model counts (MFU keeps the model-FLOPs convention).
    recompute = list(plan.get("recompute") or [])
    rec_share = recompute_flops_per_token(shape, seq, recompute) / fl["total"] if recompute and fl["total"] else 0.0
    # SMs taken by an SM-based dispatcher whose all-to-all overlaps compute.
    sm_share = 0.0
    if (str(plan.get("dispatcher", "")).lower() in SM_DISPATCHERS and plan.get("a2a_overlap")
            and int(plan.get("ep", 1)) > 1 and shape["moe_layers"]):
        sms = SMS_PER_GPU.get(hw["gpu"], 132)
        sm_share = min(0.5, float(plan.get("dispatcher_sms", COMM_SMS)) / sms)
    t_compute = t_compute * (1 + rec_share) / (1 - sm_share)
    bubble = (pp - 1) / (vp * m + pp - 1)
    t_pipe = t_compute / (1 - bubble)

    d, layers = shape["d"], shape["layers"]
    layers_stage = layers / pp
    moe_stage = shape["moe_layers"] / pp
    tokens_mb = mbs * seq / cp  # tokens of one micro-batch on one CP rank

    def exposed(t, overlap):
        return max(t * (1 - overlap), t - t_compute) if t > 0 else 0.0

    # TP with sequence parallel: per layer per micro-batch 4 all-gathers + 4 reduce-scatters
    # (fwd 2+2, bwd 2+2), each moving (tp-1)/tp of the micro-batch activations.
    t_tp = 0.0
    if tp > 1:
        b = 8 * (tp - 1) / tp * tokens_mb * d * ACT_BYTES * layers_stage * m
        t_tp = b / _span_bw(hw, 1, tp)
    # CP ring attention: per full-attention layer per micro-batch, K and V chunks pass cp-1
    # times forward and twice that backward (3 x total).
    t_cp = 0.0
    a = shape["attention"]
    if cp > 1:
        kv = tokens_mb * 2 * a["kv_heads"] * a["d_qk"] * ACT_BYTES
        b = 3 * (cp - 1) * kv * (a["full_layers"] / pp) * m
        t_cp = b / _span_bw(hw, tp, cp)
    # EP all-to-all: per MoE layer per micro-batch, dispatch + combine forward and backward
    # (4 x), of the rank's tokens x top-k x hidden, (ep-1)/ep of it leaving the rank.
    t_ep = 0.0
    if ep > 1 and shape["moe_layers"]:
        per_copy = 4 * (tokens_mb / tp) * d * ACT_BYTES * (ep - 1) / ep * moe_stage * m
        on_node = max(1, hw["gpn"] // etp)
        off = max(0.0, 1 - on_node / ep)
        # Group-limited routing (config n_group > 1, topk_group): a token reaches at most
        # topk_group groups, so at most that many copies cross the network.
        # Only a dispatcher that de-duplicates per node (DeepEP; plan dedup_by_group) uses it.
        grouped = shape["n_group"] > 1 and plan.get("dedup_by_group")
        remote = min(shape["topk"], shape["topk_group"]) if grouped else shape["topk"]
        t_ep = max(per_copy * remote * off / _bw(hw, "net"), per_copy * shape["topk"] * (1 - off) / _bw(hw, "nvlink"))
    # DP: reduce-scatter grads + all-gather params over the DP group, (g-1)/g of the local
    # shard each. Dense params: group DP x CP. Expert params: group DP x CP / EP, shard / EP.
    expert_p = shape["expert_total_b"] * 1e9
    dense_p = shape["total_b"] * 1e9 - expert_p
    local_dense = dense_p / (tp * pp)
    local_expert = expert_p / (etp * pp * max(ep, 1))
    g_dense, g_expert = dp * cp, max(1, per_stage // (etp * max(ep, 1)))
    per_byte = GRAD_BYTES + PARAM_BYTES

    def dp_time(local, g):
        if g <= 1:
            return 0.0
        return (g - 1) / g * local * per_byte / _span_bw(hw, tp, g)

    t_dp = dp_time(local_dense, g_dense) + dp_time(local_expert, g_expert)
    zero = 3 if plan.get("fsdp") else int(plan.get("zero", 1))
    if zero >= 3:
        # ZeRO-3 = FSDP full shard: per layer per micro-batch, params all-gathered before forward and
        # again before backward, grads reduce-scattered after backward (3 Psi per micro-batch against
        # 2 Psi per step for ZeRO-1/2; ZeRO paper sec. 7.2.2). Overlap is capped per layer.
        k = min(1.0, max(0.0, float(plan.get("zero3_prefetch", 1.0))))
        t_dp, e_dp = _sharded_dp_time(shape, hw, seq, t_compute, m, tp, pp, etp, ep, g_dense, g_expert, k)
    else:
        if zero == 2:
            t_dp = t_dp / 2 * (1 + m)  # grads reduce-scattered every micro-batch, params once
        e_dp = exposed(t_dp, OVERLAP["dp"])
    ov_ep = float(plan.get("overlap_ep", OVERLAP["ep"]))  # a schedule such as DualPipe hides more
    e_tp, e_cp, e_ep = exposed(t_tp, OVERLAP["tp"]), exposed(t_cp, OVERLAP["cp"]), exposed(t_ep, ov_ep)
    t_step = t_pipe + e_tp + e_cp + e_ep + e_dp
    mfu = f_gpu / (peak * t_step)
    return {
        "model_version": MODEL_VERSION,
        "precision": prec,
        "peak_tflops_per_gpu": hw[prec],
        "plan": {"num_gpus": n, "tp": tp, "pp": pp, "cp": cp, "vp": vp, "ep": ep, "etp": etp, "dp": dp, "mbs": mbs,
                 "gbs": gbs, "gbs_source": "request" if gbs_given else "default_16M_tokens",
                 "micro_batches": round(m, 2), "tokens_per_step": tokens_step, "zero": zero,
                 **({"zero3_prefetch": k} if zero >= 3 else {})},
        "flops_per_token": fl["total"],
        "attention_flops_share": round(fl["attention"] / fl["total"], 4) if fl["total"] else 0,
        "expert_flops_share": round(f_e, 4),
        "base_efficiency": round(eta_base, 4),
        "tgp_gain": round(tgp_gain, 4),
        "moe_expert_factor": round(moe_factor, 4) if f_e else None,
        "compute_efficiency": round(eta, 4),
        "recompute_flops_share": round(rec_share, 4),
        "comm_sm_share": round(sm_share, 4),
        "bubble_fraction": round(bubble, 4),
        "time_s": {"compute": t_compute, "bubble": t_pipe - t_compute, "tp_exposed": e_tp, "cp_exposed": e_cp,
                   "ep_exposed": e_ep, "dp_exposed": e_dp, "tp_total": t_tp, "cp_total": t_cp,
                   "ep_total": t_ep, "dp_total": t_dp, "step": t_step},
        "mfu": mfu,
        "model_tflops_per_gpu": f_gpu / t_step / 1e12,
        "tokens_per_second": tokens_step / t_step,
        "tokens_per_second_per_gpu": tokens_step / t_step / n,
    }


def mfu_factors(est):
    """MFU as a product: base x MoE x (1 - bubble) x compute / (compute + bubble + exposed)."""
    t = est["time_s"]
    exposed = t["tp_exposed"] + t["cp_exposed"] + t["ep_exposed"] + t["dp_exposed"]
    return {
        "compute_efficiency": est["compute_efficiency"],
        "pipeline_factor": round(1 - est["bubble_fraction"], 4),
        "communication_factor": round((t["compute"] + t["bubble"]) / t["step"], 4),
        "exposed_comm_share_of_step": round(exposed / t["step"], 4),
        "mfu": est["mfu"],
    }


def candidate_plans(num_gpus, layers, n_experts, gpn=8, seq=4096, tp_options=(1, 2, 4, 8)):
    """Plans the search tries: TP in {1,2,4,8} (<= node), PP up to 16 (layer divisors and powers
    of 2), CP 1 (CP > 1 only for sequences >= 32K), EP powers of 2 dividing the experts.
    tp_options: Trainium plans TP in NeuronCores, up to 32 (Neuron recipes: Llama 3.1 TP 32)."""
    tps = [t for t in tp_options if t <= gpn and num_gpus % t == 0]
    divisors = {p for p in range(1, layers + 1) if layers % p == 0}
    pps = sorted({p for p in divisors | {1, 2, 4, 8, 16} if p <= min(16, layers)})
    cps = [1] + ([2, 4, 8] if seq >= 32768 else [])
    eps = [1] + [e for e in (2, 4, 8, 16, 32, 64, 128, 256) if n_experts and n_experts % e == 0]
    out = []
    for tp in tps:
        for pp in pps:
            for cp in cps:
                if num_gpus % (tp * pp * cp):
                    continue
                dp = num_gpus // (tp * pp * cp)
                for ep in eps:
                    if ep > 1 and (num_gpus // pp) % ep:
                        continue
                    out.append({"tp": tp, "pp": pp, "cp": cp, "ep": ep, "dp": dp})
    return out
