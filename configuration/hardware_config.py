"""
Hardware platform configurations.

Edit this file to add or modify hardware platform specifications.
Each hardware configuration defines GPU cluster specifications including
memory, bandwidth, and compute capabilities.
"""

HARDWARE = [
    {
        "name": "128 p5en",
        "gpus": 1024,
        "nodes": 128,
        "mem_gb": 141,
        "label": "1024_H200",
        "inter_node_bw_gb": 40,  # I0nter-node network bandwidth in GB/s 
        "intra_node_bw_gbps": 900,  # Intra-node NVLink bandwidth in GB/s
        "peak_tflops_bf16": 989,  # Peak TFLOPs for BF16 precision
    },
    {
        "name": "256 p5en",
        "gpus": 2048,
        "nodes": 256,
        "mem_gb": 141,
        "label": "2048_H200",
        "inter_node_bw_gb": 40,  # Inter-node network bandwidth in GB/s 
        "intra_node_bw_gbps": 900,  # Intra-node NVLink bandwidth in GB/s
        "peak_tflops_bf16": 989,  # Peak TFLOPs for BF16 precision
    },
    {
        "name": "128 p5",
        "gpus": 1024,
        "nodes": 128,
        "mem_gb": 80,
        "label": "1024_H100",
        "inter_node_bw_gb": 25,  # I0nter-node network bandwidth in GB/s 
        "intra_node_bw_gbps": 900,  # Intra-node NVLink bandwidth in GB/s
        "peak_tflops_bf16": 989,  # Peak TFLOPs for BF16 precision
    },
    {
        "name": "256 p5",
        "gpus": 2048,
        "nodes": 256,
        "mem_gb": 80,
        "label": "2048_H100",
        "inter_node_bw_gb": 25,  # Inter-node network bandwidth in GB/s 
        "intra_node_bw_gbps": 900,  # Intra-node NVLink bandwidth in GB/s
        "peak_tflops_bf16": 989,  # Peak TFLOPs for BF16 precision
    }
]
