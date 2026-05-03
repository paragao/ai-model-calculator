"""
Hardware platform configurations.

Edit this file to add or modify hardware platform specifications.
Each hardware configuration defines GPU cluster specifications including
memory, bandwidth, and compute capabilities.

Bharat Gen - p5en (H200 141GB) and p5 (H100 80GB) configurations.
- p5.48xlarge: 8x H100 SXM, 32x100Gbps EFA NICs = 3200 Gbps = 400 GB/s
- p5en.48xlarge: 8x H200 SXM, 16x200Gbps EFA NICs = 3200 Gbps = 400 GB/s
"""

HARDWARE = [
    # p5en.48xlarge (H200 SXM, 141 GB)
    {
        "name": "128 p5en",
        "gpus": 1024,
        "nodes": 128,
        "mem_gb": 141,
        "label": "1024_H200",
        "inter_node_bw_gb": 400,
        "intra_node_bw_gbps": 900,
        "peak_tflops_bf16": 989,
    },
    {
        "name": "192 p5en",
        "gpus": 1536,
        "nodes": 192,
        "mem_gb": 141,
        "label": "1536_H200",
        "inter_node_bw_gb": 400,
        "intra_node_bw_gbps": 900,
        "peak_tflops_bf16": 989,
    },
    {
        "name": "216 p5en",
        "gpus": 1728,
        "nodes": 216,
        "mem_gb": 141,
        "label": "1728_H200",
        "inter_node_bw_gb": 400,
        "intra_node_bw_gbps": 900,
        "peak_tflops_bf16": 989,
    },
    {
        "name": "232 p5en",
        "gpus": 1856,
        "nodes": 232,
        "mem_gb": 141,
        "label": "1856_H200",
        "inter_node_bw_gb": 400,
        "intra_node_bw_gbps": 900,
        "peak_tflops_bf16": 989,
    },
    # p5.48xlarge (H100 SXM, 80 GB)
    {
        "name": "128 p5",
        "gpus": 1024,
        "nodes": 128,
        "mem_gb": 80,
        "label": "1024_H100",
        "inter_node_bw_gb": 400,
        "intra_node_bw_gbps": 900,
        "peak_tflops_bf16": 989,
    },
    {
        "name": "192 p5",
        "gpus": 1536,
        "nodes": 192,
        "mem_gb": 80,
        "label": "1536_H100",
        "inter_node_bw_gb": 400,
        "intra_node_bw_gbps": 900,
        "peak_tflops_bf16": 989,
    },
    {
        "name": "216 p5",
        "gpus": 1728,
        "nodes": 216,
        "mem_gb": 80,
        "label": "1728_H100",
        "inter_node_bw_gb": 400,
        "intra_node_bw_gbps": 900,
        "peak_tflops_bf16": 989,
    },
    {
        "name": "232 p5",
        "gpus": 1856,
        "nodes": 232,
        "mem_gb": 80,
        "label": "1856_H100",
        "inter_node_bw_gb": 400,
        "intra_node_bw_gbps": 900,
        "peak_tflops_bf16": 989,
    },
]
