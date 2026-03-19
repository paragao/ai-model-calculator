"""
Hardware platform configurations.

Edit this file to add or modify hardware platform specifications.
Each hardware configuration defines GPU cluster specifications including
memory, bandwidth, and compute capabilities.
"""

HARDWARE = [
    {
        "name": "4,096 H200",
        "gpus": 4096,
        "nodes": 512,
        "mem_gb": 141,
        "label": "4096_H200",
        "inter_node_bw_gb": 40,  # Inter-node network bandwidth in GB/s (e.g., InfiniBand HDR200)
        "intra_node_bw_gbps": 900,  # Intra-node NVLink bandwidth in GB/s
        "peak_tflops_bf16": 989,  # Peak TFLOPs for BF16 precision
    },
    {
        "name": "4,096 H100",
        "gpus": 4096,
        "nodes": 512,
        "mem_gb": 80,
        "label": "4096_H100",
        "inter_node_bw_gb": 25,  # Lower inter-node bandwidth than H200
        "intra_node_bw_gbps": 400,  # Lower NVLink bandwidth than H200
        "peak_tflops_bf16": 989,  # Peak TFLOPs for BF16 precision
    },
    {
        "name": "2,048 H200",
        "gpus": 2048,
        "nodes": 256,
        "mem_gb": 141,
        "label": "2048_H200",
        "inter_node_bw_gb": 40,
        "intra_node_bw_gbps": 900,
        "peak_tflops_bf16": 989,
    },
    {
        "name": "2,048 H100",
        "gpus": 2048,
        "nodes": 256,
        "mem_gb": 80,
        "label": "2048_H100",
        "inter_node_bw_gb": 25,
        "intra_node_bw_gbps": 400,
        "peak_tflops_bf16": 989,
    },
]
