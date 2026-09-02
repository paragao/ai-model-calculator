"""
Hardware platform configurations.

Pre-defined AWS EC2 GPU instance types at various cluster sizes.
Each hardware configuration defines GPU cluster specifications including
memory, bandwidth, and compute capabilities.

Instance specifications:
- p5.48xlarge:        8x H100 SXM 80GB,   3200 Gbps EFA  = 400 GB/s, 900 GB/s NVLink, 989 BF16 TFLOPS/GPU
- p5en.48xlarge:     8x H200 SXM 141GB,  3200 Gbps EFAv3 = 400 GB/s, 900 GB/s NVLink, 989 BF16 TFLOPS/GPU
- p6-b200.48xlarge:  8x B200 180GB,      3200 Gbps EFAv4 = 400 GB/s, 1800 GB/s NVLink5, 2250 BF16 TFLOPS/GPU
- p6-b300.48xlarge:  8x B300 Ultra 268GB, 6400 Gbps EFAv4 = 800 GB/s, 1800 GB/s NVLink5, 3375 BF16 TFLOPS/GPU
- p6e-gb200.36xlarge: 4x GB200 NVL 185GB, 3200 Gbps EFAv4 = 400 GB/s, 900 GB/s NVLink, 2500 BF16 TFLOPS/GPU
- g5.48xlarge:       8x A10G 24GB,       100 Gbps ENA   = 100 GB/s, 600 GB/s NVLink, 35 BF16 TFLOPS/GPU
- g6.48xlarge:       8x L4 24GB,         100 Gbps ENA   = 100 GB/s, 64 GB/s PCIe, 121 BF16 TFLOPS/GPU
- g6e.48xlarge:      8x L40S 48GB,       400 Gbps ENA   = 50 GB/s, 64 GB/s PCIe, 366 BF16 TFLOPS/GPU
- g7e.48xlarge:      8x L40S v2 48GB,    400 Gbps EFA   = 50 GB/s, 128 GB/s PCIe5, 366 BF16 TFLOPS/GPU
- g7e.12xlarge:      2x RTX PRO 6000 96GB, 400 Gbps EFA = 50 GB/s, 128 GB/s PCIe5, 1044 BF16 TFLOPS/GPU

Inference-relevant fields:
  hbm_bw_gbps:       HBM bandwidth per GPU (GB/s) - CRITICAL for decode phase
  pcie_bw_gbps:      PCIe bandwidth (GB/s) - used for TP over PCIe (G-family)
  gpu_arch:          GPU architecture generation
  fp8_tflops:        FP8 peak TFLOPS/GPU (for prefill compute bound calculations)
  instance_type:     EC2 instance type (for cost calculations)
  gpus_per_node:     GPUs per physical node
"""

HARDWARE = [
    # =============================================
    # p5.48xlarge (H100 SXM, 80 GB)
    # =============================================
    {"name": "64 p5", "gpus": 512, "nodes": 64, "mem_gb": 80, "label": "512_H100", "inter_node_bw_gb": 400, "intra_node_bw_gbps": 900, "peak_tflops_bf16": 989, "hbm_bw_gbps": 3350, "pcie_bw_gbps": 64, "gpu_arch": "hopper", "fp8_tflops": 1979, "instance_type": "p5.48xlarge", "gpus_per_node": 8},
    {"name": "128 p5", "gpus": 1024, "nodes": 128, "mem_gb": 80, "label": "1024_H100", "inter_node_bw_gb": 400, "intra_node_bw_gbps": 900, "peak_tflops_bf16": 989, "hbm_bw_gbps": 3350, "pcie_bw_gbps": 64, "gpu_arch": "hopper", "fp8_tflops": 1979, "instance_type": "p5.48xlarge", "gpus_per_node": 8},
    {"name": "256 p5", "gpus": 2048, "nodes": 256, "mem_gb": 80, "label": "2048_H100", "inter_node_bw_gb": 400, "intra_node_bw_gbps": 900, "peak_tflops_bf16": 989, "hbm_bw_gbps": 3350, "pcie_bw_gbps": 64, "gpu_arch": "hopper", "fp8_tflops": 1979, "instance_type": "p5.48xlarge", "gpus_per_node": 8},
    {"name": "512 p5", "gpus": 4096, "nodes": 512, "mem_gb": 80, "label": "4096_H100", "inter_node_bw_gb": 400, "intra_node_bw_gbps": 900, "peak_tflops_bf16": 989, "hbm_bw_gbps": 3350, "pcie_bw_gbps": 64, "gpu_arch": "hopper", "fp8_tflops": 1979, "instance_type": "p5.48xlarge", "gpus_per_node": 8},
    # =============================================
    # p5en.48xlarge (H200 SXM, 141 GB)
    # =============================================
    {"name": "64 p5en", "gpus": 512, "nodes": 64, "mem_gb": 141, "label": "512_H200", "inter_node_bw_gb": 400, "intra_node_bw_gbps": 900, "peak_tflops_bf16": 989, "hbm_bw_gbps": 4800, "pcie_bw_gbps": 64, "gpu_arch": "hopper", "fp8_tflops": 1979, "instance_type": "p5en.48xlarge", "gpus_per_node": 8},
    {"name": "128 p5en", "gpus": 1024, "nodes": 128, "mem_gb": 141, "label": "1024_H200", "inter_node_bw_gb": 400, "intra_node_bw_gbps": 900, "peak_tflops_bf16": 989, "hbm_bw_gbps": 4800, "pcie_bw_gbps": 64, "gpu_arch": "hopper", "fp8_tflops": 1979, "instance_type": "p5en.48xlarge", "gpus_per_node": 8},
    {"name": "256 p5en", "gpus": 2048, "nodes": 256, "mem_gb": 141, "label": "2048_H200", "inter_node_bw_gb": 400, "intra_node_bw_gbps": 900, "peak_tflops_bf16": 989, "hbm_bw_gbps": 4800, "pcie_bw_gbps": 64, "gpu_arch": "hopper", "fp8_tflops": 1979, "instance_type": "p5en.48xlarge", "gpus_per_node": 8},
    {"name": "512 p5en", "gpus": 4096, "nodes": 512, "mem_gb": 141, "label": "4096_H200", "inter_node_bw_gb": 400, "intra_node_bw_gbps": 900, "peak_tflops_bf16": 989, "hbm_bw_gbps": 4800, "pcie_bw_gbps": 64, "gpu_arch": "hopper", "fp8_tflops": 1979, "instance_type": "p5en.48xlarge", "gpus_per_node": 8},
    # =============================================
    # p6-b200.48xlarge (B200, 180 GB)
    # =============================================
    {"name": "64 p6-b200", "gpus": 512, "nodes": 64, "mem_gb": 180, "label": "512_B200", "inter_node_bw_gb": 400, "intra_node_bw_gbps": 1800, "peak_tflops_bf16": 2250, "hbm_bw_gbps": 8000, "pcie_bw_gbps": 128, "gpu_arch": "blackwell", "fp8_tflops": 4500, "instance_type": "p6-b200.48xlarge", "gpus_per_node": 8},
    {"name": "128 p6-b200", "gpus": 1024, "nodes": 128, "mem_gb": 180, "label": "1024_B200", "inter_node_bw_gb": 400, "intra_node_bw_gbps": 1800, "peak_tflops_bf16": 2250, "hbm_bw_gbps": 8000, "pcie_bw_gbps": 128, "gpu_arch": "blackwell", "fp8_tflops": 4500, "instance_type": "p6-b200.48xlarge", "gpus_per_node": 8},
    {"name": "256 p6-b200", "gpus": 2048, "nodes": 256, "mem_gb": 180, "label": "2048_B200", "inter_node_bw_gb": 400, "intra_node_bw_gbps": 1800, "peak_tflops_bf16": 2250, "hbm_bw_gbps": 8000, "pcie_bw_gbps": 128, "gpu_arch": "blackwell", "fp8_tflops": 4500, "instance_type": "p6-b200.48xlarge", "gpus_per_node": 8},
    {"name": "512 p6-b200", "gpus": 4096, "nodes": 512, "mem_gb": 180, "label": "4096_B200", "inter_node_bw_gb": 400, "intra_node_bw_gbps": 1800, "peak_tflops_bf16": 2250, "hbm_bw_gbps": 8000, "pcie_bw_gbps": 128, "gpu_arch": "blackwell", "fp8_tflops": 4500, "instance_type": "p6-b200.48xlarge", "gpus_per_node": 8},
    # =============================================
    # p6-b300.48xlarge (B300 Ultra, 268 GB)
    # =============================================
    {"name": "64 p6-b300", "gpus": 512, "nodes": 64, "mem_gb": 268, "label": "512_B300", "inter_node_bw_gb": 800, "intra_node_bw_gbps": 1800, "peak_tflops_bf16": 3375, "hbm_bw_gbps": 12000, "pcie_bw_gbps": 128, "gpu_arch": "blackwell", "fp8_tflops": 6750, "instance_type": "p6-b300.48xlarge", "gpus_per_node": 8},
    {"name": "128 p6-b300", "gpus": 1024, "nodes": 128, "mem_gb": 268, "label": "1024_B300", "inter_node_bw_gb": 800, "intra_node_bw_gbps": 1800, "peak_tflops_bf16": 3375, "hbm_bw_gbps": 12000, "pcie_bw_gbps": 128, "gpu_arch": "blackwell", "fp8_tflops": 6750, "instance_type": "p6-b300.48xlarge", "gpus_per_node": 8},
    {"name": "256 p6-b300", "gpus": 2048, "nodes": 256, "mem_gb": 268, "label": "2048_B300", "inter_node_bw_gb": 800, "intra_node_bw_gbps": 1800, "peak_tflops_bf16": 3375, "hbm_bw_gbps": 12000, "pcie_bw_gbps": 128, "gpu_arch": "blackwell", "fp8_tflops": 6750, "instance_type": "p6-b300.48xlarge", "gpus_per_node": 8},
    {"name": "512 p6-b300", "gpus": 4096, "nodes": 512, "mem_gb": 268, "label": "4096_B300", "inter_node_bw_gb": 800, "intra_node_bw_gbps": 1800, "peak_tflops_bf16": 3375, "hbm_bw_gbps": 12000, "pcie_bw_gbps": 128, "gpu_arch": "blackwell", "fp8_tflops": 6750, "instance_type": "p6-b300.48xlarge", "gpus_per_node": 8},
    # =============================================
    # g5.48xlarge (A10G, 24 GB)
    # =============================================
    {"name": "1 g5", "gpus": 8, "nodes": 1, "mem_gb": 24, "label": "8_A10G", "inter_node_bw_gb": 100, "intra_node_bw_gbps": 600, "peak_tflops_bf16": 35, "hbm_bw_gbps": 600, "pcie_bw_gbps": 32, "gpu_arch": "ampere", "fp8_tflops": 0, "instance_type": "g5.48xlarge", "gpus_per_node": 8},
    {"name": "4 g5", "gpus": 32, "nodes": 4, "mem_gb": 24, "label": "32_A10G", "inter_node_bw_gb": 100, "intra_node_bw_gbps": 600, "peak_tflops_bf16": 35, "hbm_bw_gbps": 600, "pcie_bw_gbps": 32, "gpu_arch": "ampere", "fp8_tflops": 0, "instance_type": "g5.48xlarge", "gpus_per_node": 8},
    {"name": "8 g5", "gpus": 64, "nodes": 8, "mem_gb": 24, "label": "64_A10G", "inter_node_bw_gb": 100, "intra_node_bw_gbps": 600, "peak_tflops_bf16": 35, "hbm_bw_gbps": 600, "pcie_bw_gbps": 32, "gpu_arch": "ampere", "fp8_tflops": 0, "instance_type": "g5.48xlarge", "gpus_per_node": 8},
    {"name": "16 g5", "gpus": 128, "nodes": 16, "mem_gb": 24, "label": "128_A10G", "inter_node_bw_gb": 100, "intra_node_bw_gbps": 600, "peak_tflops_bf16": 35, "hbm_bw_gbps": 600, "pcie_bw_gbps": 32, "gpu_arch": "ampere", "fp8_tflops": 0, "instance_type": "g5.48xlarge", "gpus_per_node": 8},
    # =============================================
    # g6.48xlarge (L4, 24 GB)
    # =============================================
    {"name": "1 g6", "gpus": 8, "nodes": 1, "mem_gb": 24, "label": "8_L4", "inter_node_bw_gb": 100, "intra_node_bw_gbps": 64, "peak_tflops_bf16": 121, "hbm_bw_gbps": 300, "pcie_bw_gbps": 32, "gpu_arch": "ada", "fp8_tflops": 242, "instance_type": "g6.48xlarge", "gpus_per_node": 8},
    {"name": "4 g6", "gpus": 32, "nodes": 4, "mem_gb": 24, "label": "32_L4", "inter_node_bw_gb": 100, "intra_node_bw_gbps": 64, "peak_tflops_bf16": 121, "hbm_bw_gbps": 300, "pcie_bw_gbps": 32, "gpu_arch": "ada", "fp8_tflops": 242, "instance_type": "g6.48xlarge", "gpus_per_node": 8},
    {"name": "8 g6", "gpus": 64, "nodes": 8, "mem_gb": 24, "label": "64_L4", "inter_node_bw_gb": 100, "intra_node_bw_gbps": 64, "peak_tflops_bf16": 121, "hbm_bw_gbps": 300, "pcie_bw_gbps": 32, "gpu_arch": "ada", "fp8_tflops": 242, "instance_type": "g6.48xlarge", "gpus_per_node": 8},
    {"name": "16 g6", "gpus": 128, "nodes": 16, "mem_gb": 24, "label": "128_L4", "inter_node_bw_gb": 100, "intra_node_bw_gbps": 64, "peak_tflops_bf16": 121, "hbm_bw_gbps": 300, "pcie_bw_gbps": 32, "gpu_arch": "ada", "fp8_tflops": 242, "instance_type": "g6.48xlarge", "gpus_per_node": 8},
    # =============================================
    # g6e.48xlarge (L40S, 48 GB)
    # =============================================
    {"name": "1 g6e", "gpus": 8, "nodes": 1, "mem_gb": 48, "label": "8_L40S", "inter_node_bw_gb": 50, "intra_node_bw_gbps": 64, "peak_tflops_bf16": 366, "hbm_bw_gbps": 864, "pcie_bw_gbps": 64, "gpu_arch": "ada", "fp8_tflops": 733, "instance_type": "g6e.48xlarge", "gpus_per_node": 8},
    {"name": "4 g6e", "gpus": 32, "nodes": 4, "mem_gb": 48, "label": "32_L40S", "inter_node_bw_gb": 50, "intra_node_bw_gbps": 64, "peak_tflops_bf16": 366, "hbm_bw_gbps": 864, "pcie_bw_gbps": 64, "gpu_arch": "ada", "fp8_tflops": 733, "instance_type": "g6e.48xlarge", "gpus_per_node": 8},
    {"name": "8 g6e", "gpus": 64, "nodes": 8, "mem_gb": 48, "label": "64_L40S", "inter_node_bw_gb": 50, "intra_node_bw_gbps": 64, "peak_tflops_bf16": 366, "hbm_bw_gbps": 864, "pcie_bw_gbps": 64, "gpu_arch": "ada", "fp8_tflops": 733, "instance_type": "g6e.48xlarge", "gpus_per_node": 8},
    {"name": "16 g6e", "gpus": 128, "nodes": 16, "mem_gb": 48, "label": "128_L40S", "inter_node_bw_gb": 50, "intra_node_bw_gbps": 64, "peak_tflops_bf16": 366, "hbm_bw_gbps": 864, "pcie_bw_gbps": 64, "gpu_arch": "ada", "fp8_tflops": 733, "instance_type": "g6e.48xlarge", "gpus_per_node": 8},
]


# ============================================================================
# INFERENCE HARDWARE: Single-node configurations for inference workloads
# These use fewer nodes (1-2) and may use GPU subsets (TP < gpus_per_node)
# ============================================================================

INFERENCE_HARDWARE = [
    # P-family single-node inference
    {"name": "1 p5en", "gpus": 8, "nodes": 1, "mem_gb": 141, "label": "8_H200", "inter_node_bw_gb": 400, "intra_node_bw_gbps": 900, "peak_tflops_bf16": 989, "hbm_bw_gbps": 4800, "pcie_bw_gbps": 64, "gpu_arch": "hopper", "fp8_tflops": 1979, "instance_type": "p5en.48xlarge", "gpus_per_node": 8},
    {"name": "1 p6-b200", "gpus": 8, "nodes": 1, "mem_gb": 180, "label": "8_B200", "inter_node_bw_gb": 400, "intra_node_bw_gbps": 1800, "peak_tflops_bf16": 2250, "hbm_bw_gbps": 8000, "pcie_bw_gbps": 128, "gpu_arch": "blackwell", "fp8_tflops": 4500, "instance_type": "p6-b200.48xlarge", "gpus_per_node": 8},
    {"name": "1 p6-b300", "gpus": 8, "nodes": 1, "mem_gb": 268, "label": "8_B300", "inter_node_bw_gb": 800, "intra_node_bw_gbps": 1800, "peak_tflops_bf16": 3375, "hbm_bw_gbps": 12000, "pcie_bw_gbps": 128, "gpu_arch": "blackwell", "fp8_tflops": 6750, "instance_type": "p6-b300.48xlarge", "gpus_per_node": 8},
    # G6e single-node inference
    {"name": "1 g6e-48xl", "gpus": 8, "nodes": 1, "mem_gb": 48, "label": "8_L40S", "inter_node_bw_gb": 50, "intra_node_bw_gbps": 64, "peak_tflops_bf16": 366, "hbm_bw_gbps": 864, "pcie_bw_gbps": 64, "gpu_arch": "ada", "fp8_tflops": 733, "instance_type": "g6e.48xlarge", "gpus_per_node": 8},
    # G7e inference (Blackwell RTX PRO)
    {"name": "1 g7e-48xl", "gpus": 8, "nodes": 1, "mem_gb": 48, "label": "8_L40Sv2", "inter_node_bw_gb": 50, "intra_node_bw_gbps": 128, "peak_tflops_bf16": 366, "hbm_bw_gbps": 1500, "pcie_bw_gbps": 128, "gpu_arch": "blackwell", "fp8_tflops": 733, "instance_type": "g7e.48xlarge", "gpus_per_node": 8},
    {"name": "1 g7e-12xl", "gpus": 2, "nodes": 1, "mem_gb": 96, "label": "2_RTX_PRO_6000", "inter_node_bw_gb": 50, "intra_node_bw_gbps": 128, "peak_tflops_bf16": 1044, "hbm_bw_gbps": 1750, "pcie_bw_gbps": 128, "gpu_arch": "blackwell", "fp8_tflops": 2088, "instance_type": "g7e.12xlarge", "gpus_per_node": 2},
]
