"""
Hardware platform configurations.

Pre-defined AWS EC2 GPU instance types at various cluster sizes.
Each hardware configuration defines GPU cluster specifications including
memory, bandwidth, and compute capabilities.

Instance specifications:
- p5.48xlarge:        8x H100 SXM 80GB,   3200 Gbps EFA  = 400 GB/s, 900 GB/s NVLink, 989 BF16 TFLOPS/GPU
- p5en.48xlarge:     8x H200 SXM 141GB,  3200 Gbps EFAv3 = 400 GB/s, 900 GB/s NVLink, 989 BF16 TFLOPS/GPU
- p6-b200.48xlarge:  8x B200 180GB,      3200 Gbps EFAv4 = 400 GB/s, 1800 GB/s NVLink5, 2250 BF16 TFLOPS/GPU
- p6-b300.48xlarge:  8x B300 Ultra 268GB, 6400 Gbps EFAv4 = 800 GB/s, 1800 GB/s NVLink5, 2250 BF16 TFLOPS/GPU
- p6e-gb200.36xlarge: 4x GB200 NVL 185GB, 3200 Gbps EFAv4 = 400 GB/s, 900 GB/s NVLink, 2500 BF16 TFLOPS/GPU
  (reference specs only -- not present in HARDWARE below; pass it via --hardware)
- g5.48xlarge:       8x A10G 24GB,       100 Gbps ENA   = 12.5 GB/s, 64 GB/s PCIe, 35 BF16 TFLOPS/GPU
- g6.48xlarge:       8x L4 24GB,         100 Gbps ENA   = 12.5 GB/s, 64 GB/s PCIe, 121 BF16 TFLOPS/GPU
- g6e.48xlarge:      8x L40S 48GB,       400 Gbps ENA   = 50 GB/s, 64 GB/s PCIe, 366 BF16 TFLOPS/GPU

Units: `inter_node_bw_gb` and `intra_node_bw_gbps` are BOTH in GB/s despite the
differing suffixes. Divide an advertised Gbps figure by 8. NVLink figures are
the bidirectional totals (H100/H200 900 GB/s, Blackwell NVLink5 1800 GB/s).

Notes on values that are easy to get wrong:
- **B300 BF16 is 2250, the SAME as B200, not 1.5x it.** Blackwell Ultra's 1.5x
  uplift is FP4 only -- FP8 and BF16 stay at B200 levels. AWS's own announcement
  qualifies it as "1.5x GPU TFLOPS (at FP4, without sparsity)". Applying that
  ratio to BF16 overstates BF16 training throughput by 50%.
  https://aws.amazon.com/about-aws/whats-new/2025/11/amazon-ec2-p6-b300-instances-nvidia-blackwell-ultra-gpus-available/
- **B300 memory is 268 GB usable** of 288 GB raw, consistent with AWS's stated
  "2.1 TB high bandwidth GPU memory" across 8 GPUs (8 x 268 = 2.14 TB).
- **A10G, L4 and L40S have NO NVLink.** The G-family GPUs are PCIe pass-through,
  so intra-node bandwidth is PCIe-class (~64 GB/s), not an NVLink figure.
- G-family entries assume the .48xlarge size (8 GPUs). Smaller sizes have 1 or 4
  GPUs and different network bandwidth, so supply those via --hardware.
- All BF16 figures are DENSE, without sparsity. Sparsity doubles the marketing
  number and must not be used here.
"""

HARDWARE = [
    # =============================================
    # p5.48xlarge (H100 SXM, 80 GB)
    # =============================================
    {"name": "64 p5", "gpus": 512, "nodes": 64, "mem_gb": 80, "label": "512_H100", "inter_node_bw_gb": 400, "intra_node_bw_gbps": 900, "peak_tflops_bf16": 989},
    {"name": "128 p5", "gpus": 1024, "nodes": 128, "mem_gb": 80, "label": "1024_H100", "inter_node_bw_gb": 400, "intra_node_bw_gbps": 900, "peak_tflops_bf16": 989},
    {"name": "256 p5", "gpus": 2048, "nodes": 256, "mem_gb": 80, "label": "2048_H100", "inter_node_bw_gb": 400, "intra_node_bw_gbps": 900, "peak_tflops_bf16": 989},
    {"name": "512 p5", "gpus": 4096, "nodes": 512, "mem_gb": 80, "label": "4096_H100", "inter_node_bw_gb": 400, "intra_node_bw_gbps": 900, "peak_tflops_bf16": 989},
    # =============================================
    # p5en.48xlarge (H200 SXM, 141 GB)
    # =============================================
    {"name": "64 p5en", "gpus": 512, "nodes": 64, "mem_gb": 141, "label": "512_H200", "inter_node_bw_gb": 400, "intra_node_bw_gbps": 900, "peak_tflops_bf16": 989},
    {"name": "128 p5en", "gpus": 1024, "nodes": 128, "mem_gb": 141, "label": "1024_H200", "inter_node_bw_gb": 400, "intra_node_bw_gbps": 900, "peak_tflops_bf16": 989},
    {"name": "256 p5en", "gpus": 2048, "nodes": 256, "mem_gb": 141, "label": "2048_H200", "inter_node_bw_gb": 400, "intra_node_bw_gbps": 900, "peak_tflops_bf16": 989},
    {"name": "512 p5en", "gpus": 4096, "nodes": 512, "mem_gb": 141, "label": "4096_H200", "inter_node_bw_gb": 400, "intra_node_bw_gbps": 900, "peak_tflops_bf16": 989},
    # =============================================
    # p6-b200.48xlarge (B200, 180 GB)
    # =============================================
    {"name": "64 p6-b200", "gpus": 512, "nodes": 64, "mem_gb": 180, "label": "512_B200", "inter_node_bw_gb": 400, "intra_node_bw_gbps": 1800, "peak_tflops_bf16": 2250},
    {"name": "128 p6-b200", "gpus": 1024, "nodes": 128, "mem_gb": 180, "label": "1024_B200", "inter_node_bw_gb": 400, "intra_node_bw_gbps": 1800, "peak_tflops_bf16": 2250},
    {"name": "256 p6-b200", "gpus": 2048, "nodes": 256, "mem_gb": 180, "label": "2048_B200", "inter_node_bw_gb": 400, "intra_node_bw_gbps": 1800, "peak_tflops_bf16": 2250},
    {"name": "512 p6-b200", "gpus": 4096, "nodes": 512, "mem_gb": 180, "label": "4096_B200", "inter_node_bw_gb": 400, "intra_node_bw_gbps": 1800, "peak_tflops_bf16": 2250},
    # =============================================
    # p6-b300.48xlarge (B300 Ultra, 268 GB)
    # =============================================
    {"name": "64 p6-b300", "gpus": 512, "nodes": 64, "mem_gb": 268, "label": "512_B300", "inter_node_bw_gb": 800, "intra_node_bw_gbps": 1800, "peak_tflops_bf16": 2250},
    {"name": "128 p6-b300", "gpus": 1024, "nodes": 128, "mem_gb": 268, "label": "1024_B300", "inter_node_bw_gb": 800, "intra_node_bw_gbps": 1800, "peak_tflops_bf16": 2250},
    {"name": "256 p6-b300", "gpus": 2048, "nodes": 256, "mem_gb": 268, "label": "2048_B300", "inter_node_bw_gb": 800, "intra_node_bw_gbps": 1800, "peak_tflops_bf16": 2250},
    {"name": "512 p6-b300", "gpus": 4096, "nodes": 512, "mem_gb": 268, "label": "4096_B300", "inter_node_bw_gb": 800, "intra_node_bw_gbps": 1800, "peak_tflops_bf16": 2250},
    # =============================================
    # g5.48xlarge (A10G, 24 GB)
    # =============================================
    {"name": "1 g5", "gpus": 8, "nodes": 1, "mem_gb": 24, "label": "8_A10G", "inter_node_bw_gb": 12.5, "intra_node_bw_gbps": 64, "peak_tflops_bf16": 35},
    {"name": "4 g5", "gpus": 32, "nodes": 4, "mem_gb": 24, "label": "32_A10G", "inter_node_bw_gb": 12.5, "intra_node_bw_gbps": 64, "peak_tflops_bf16": 35},
    {"name": "8 g5", "gpus": 64, "nodes": 8, "mem_gb": 24, "label": "64_A10G", "inter_node_bw_gb": 12.5, "intra_node_bw_gbps": 64, "peak_tflops_bf16": 35},
    {"name": "16 g5", "gpus": 128, "nodes": 16, "mem_gb": 24, "label": "128_A10G", "inter_node_bw_gb": 12.5, "intra_node_bw_gbps": 64, "peak_tflops_bf16": 35},
    # =============================================
    # g6.48xlarge (L4, 24 GB)
    # =============================================
    {"name": "1 g6", "gpus": 8, "nodes": 1, "mem_gb": 24, "label": "8_L4", "inter_node_bw_gb": 12.5, "intra_node_bw_gbps": 64, "peak_tflops_bf16": 121},
    {"name": "4 g6", "gpus": 32, "nodes": 4, "mem_gb": 24, "label": "32_L4", "inter_node_bw_gb": 12.5, "intra_node_bw_gbps": 64, "peak_tflops_bf16": 121},
    {"name": "8 g6", "gpus": 64, "nodes": 8, "mem_gb": 24, "label": "64_L4", "inter_node_bw_gb": 12.5, "intra_node_bw_gbps": 64, "peak_tflops_bf16": 121},
    {"name": "16 g6", "gpus": 128, "nodes": 16, "mem_gb": 24, "label": "128_L4", "inter_node_bw_gb": 12.5, "intra_node_bw_gbps": 64, "peak_tflops_bf16": 121},
    # =============================================
    # g6e.48xlarge (L40S, 48 GB)
    # =============================================
    {"name": "1 g6e", "gpus": 8, "nodes": 1, "mem_gb": 48, "label": "8_L40S", "inter_node_bw_gb": 50, "intra_node_bw_gbps": 64, "peak_tflops_bf16": 366},
    {"name": "4 g6e", "gpus": 32, "nodes": 4, "mem_gb": 48, "label": "32_L40S", "inter_node_bw_gb": 50, "intra_node_bw_gbps": 64, "peak_tflops_bf16": 366},
    {"name": "8 g6e", "gpus": 64, "nodes": 8, "mem_gb": 48, "label": "64_L40S", "inter_node_bw_gb": 50, "intra_node_bw_gbps": 64, "peak_tflops_bf16": 366},
    {"name": "16 g6e", "gpus": 128, "nodes": 16, "mem_gb": 48, "label": "128_L40S", "inter_node_bw_gb": 50, "intra_node_bw_gbps": 64, "peak_tflops_bf16": 366},
]
