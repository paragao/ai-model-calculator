# Hardware Catalog

## Pre-defined hardware platforms

### Training-class instances (P-family)

| Platform | GPU | GPUs/node | Memory/GPU | Inter-node BW | Intra-node BW | Peak TFLOPS (BF16) |
|----------|-----|-----------|-----------|--------------|---------------|-------------------|
| p5.48xlarge | H100 SXM | 8 | 80 GB | 400 GB/s | 900 GB/s | 989 |
| p5e.48xlarge | H200 SXM | 8 | 141 GB | 400 GB/s | 900 GB/s | 989 |
| p5en.48xlarge | H200 SXM | 8 | 141 GB | 400 GB/s | 900 GB/s | 989 |
| p6-b200.48xlarge | B200 | 8 | 180 GB | 400 GB/s | 1800 GB/s | 2250 |
| p6-b300.48xlarge | B300 Ultra | 8 | 268 GB | 800 GB/s | 1800 GB/s | 3375 |
| p6e-gb200.36xlarge | GB200 NVL | 4 | 185 GB | 400 GB/s | 900 GB/s | 2500 |

### Inference/fine-tuning instances (G-family)

| Platform | GPU | GPUs/node | Memory/GPU | Inter-node BW | Intra-node BW | Peak TFLOPS (BF16) |
|----------|-----|-----------|-----------|--------------|---------------|-------------------|
| g5.48xlarge | A10G | 8 | 24 GB | 100 GB/s | 600 GB/s | 35 |
| g6.48xlarge | L4 | 8 | 24 GB | 100 GB/s | 64 GB/s | 121 |
| g6e.48xlarge | L40S | 8 | 48 GB | 50 GB/s | 64 GB/s | 366 |

**Notes:**
- P5/P5e/P5en: 3200 Gbps EFA networking, NVSwitch intra-node
- P6-B200: 3200 Gbps EFAv4, NVLink5 intra-node
- P6-B300: 6400 Gbps EFAv4, NVLink5 intra-node
- P6e-GB200: Available only as UltraServers (36 or 72 GPU configurations)
- G5: NVLink pairs (not full mesh), ENA networking
- G6/G6e: PCIe-connected GPUs (no NVLink), ENA networking
- G-family instances are primarily for inference and small-scale fine-tuning, not large distributed training

Ask the user which platform(s) and node count(s) to analyze. They can select multiple for comparison.

## Required inputs for custom hardware

If the user selects "Custom hardware", gather these parameters:

| Parameter | Description | Example (p5en) | Required |
|-----------|-------------|----------------|----------|
| `name` | Platform label | `"512 p5en"` | Yes |
| `nodes` | Number of nodes | `512` | Yes |
| `gpus` | Total GPUs (nodes x GPUs/node) | `4096` | Yes |
| `mem_gb` | GPU memory in GB | `141` | Yes |
| `peak_tflops_bf16` | Peak BF16 TFLOPS per GPU | `989` | Yes |
| `inter_node_bw_gb` | Inter-node bandwidth (GB/s) | `400` | Yes |
| `intra_node_bw_gbps` | Intra-node NVLink bandwidth (GB/s) | `900` | Yes |

## The hardware_config.py template

Save this file to `/tmp/ai-model-calculator/configuration/hardware_config.py` before running the calculator. Include only the hardware configurations the user selected.

```python
"""
Hardware platform configurations.

Pre-defined AWS EC2 GPU instance types at various cluster sizes.
Each hardware configuration defines GPU cluster specifications including
memory, bandwidth, and compute capabilities.

Instance specifications:
- p5.48xlarge:        8x H100 SXM 80GB,   3200 Gbps EFA  = 400 GB/s, 900 GB/s NVLink, 989 BF16 TFLOPS/GPU
- p5e.48xlarge:      8x H200 SXM 141GB,  3200 Gbps EFA  = 400 GB/s, 900 GB/s NVLink, 989 BF16 TFLOPS/GPU
- p5en.48xlarge:     8x H200 SXM 141GB,  3200 Gbps EFAv3 = 400 GB/s, 900 GB/s NVLink, 989 BF16 TFLOPS/GPU
- p6-b200.48xlarge:  8x B200 180GB,      3200 Gbps EFAv4 = 400 GB/s, 1800 GB/s NVLink5, 2250 BF16 TFLOPS/GPU
- p6-b300.48xlarge:  8x B300 Ultra 268GB, 6400 Gbps EFAv4 = 800 GB/s, 1800 GB/s NVLink5, 3375 BF16 TFLOPS/GPU
- p6e-gb200.36xlarge: 4x GB200 NVL 185GB, 3200 Gbps EFAv4 = 400 GB/s, 900 GB/s NVLink, 2500 BF16 TFLOPS/GPU
- g5.48xlarge:       8x A10G 24GB,       100 Gbps ENA   = 100 GB/s, 600 GB/s NVLink, 35 BF16 TFLOPS/GPU
- g6.48xlarge:       8x L4 24GB,         100 Gbps ENA   = 100 GB/s, 64 GB/s PCIe, 121 BF16 TFLOPS/GPU
- g6e.48xlarge:      8x L40S 48GB,       400 Gbps ENA   = 50 GB/s, 64 GB/s PCIe, 366 BF16 TFLOPS/GPU
"""

# ============================================================================
# p5.48xlarge (H100 SXM, 80 GB)
# ============================================================================

P5_HARDWARE = [
    {
        "name": "64 p5",
        "gpus": 512,
        "nodes": 64,
        "mem_gb": 80,
        "label": "512_H100",
        "inter_node_bw_gb": 400,
        "intra_node_bw_gbps": 900,
        "peak_tflops_bf16": 989,
    },
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
        "name": "256 p5",
        "gpus": 2048,
        "nodes": 256,
        "mem_gb": 80,
        "label": "2048_H100",
        "inter_node_bw_gb": 400,
        "intra_node_bw_gbps": 900,
        "peak_tflops_bf16": 989,
    },
    {
        "name": "512 p5",
        "gpus": 4096,
        "nodes": 512,
        "mem_gb": 80,
        "label": "4096_H100",
        "inter_node_bw_gb": 400,
        "intra_node_bw_gbps": 900,
        "peak_tflops_bf16": 989,
    },
]

# ============================================================================
# p5en.48xlarge (H200 SXM, 141 GB)
# ============================================================================

P5EN_HARDWARE = [
    {
        "name": "64 p5en",
        "gpus": 512,
        "nodes": 64,
        "mem_gb": 141,
        "label": "512_H200",
        "inter_node_bw_gb": 400,
        "intra_node_bw_gbps": 900,
        "peak_tflops_bf16": 989,
    },
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
        "name": "256 p5en",
        "gpus": 2048,
        "nodes": 256,
        "mem_gb": 141,
        "label": "2048_H200",
        "inter_node_bw_gb": 400,
        "intra_node_bw_gbps": 900,
        "peak_tflops_bf16": 989,
    },
    {
        "name": "512 p5en",
        "gpus": 4096,
        "nodes": 512,
        "mem_gb": 141,
        "label": "4096_H200",
        "inter_node_bw_gb": 400,
        "intra_node_bw_gbps": 900,
        "peak_tflops_bf16": 989,
    },
]

# ============================================================================
# p6-b200.48xlarge (B200, 180 GB)
# ============================================================================

P6_B200_HARDWARE = [
    {
        "name": "64 p6-b200",
        "gpus": 512,
        "nodes": 64,
        "mem_gb": 180,
        "label": "512_B200",
        "inter_node_bw_gb": 400,
        "intra_node_bw_gbps": 1800,
        "peak_tflops_bf16": 2250,
    },
    {
        "name": "128 p6-b200",
        "gpus": 1024,
        "nodes": 128,
        "mem_gb": 180,
        "label": "1024_B200",
        "inter_node_bw_gb": 400,
        "intra_node_bw_gbps": 1800,
        "peak_tflops_bf16": 2250,
    },
    {
        "name": "256 p6-b200",
        "gpus": 2048,
        "nodes": 256,
        "mem_gb": 180,
        "label": "2048_B200",
        "inter_node_bw_gb": 400,
        "intra_node_bw_gbps": 1800,
        "peak_tflops_bf16": 2250,
    },
    {
        "name": "512 p6-b200",
        "gpus": 4096,
        "nodes": 512,
        "mem_gb": 180,
        "label": "4096_B200",
        "inter_node_bw_gb": 400,
        "intra_node_bw_gbps": 1800,
        "peak_tflops_bf16": 2250,
    },
]

# ============================================================================
# p6-b300.48xlarge (B300 Ultra, 268 GB)
# ============================================================================

P6_B300_HARDWARE = [
    {
        "name": "64 p6-b300",
        "gpus": 512,
        "nodes": 64,
        "mem_gb": 268,
        "label": "512_B300",
        "inter_node_bw_gb": 800,
        "intra_node_bw_gbps": 1800,
        "peak_tflops_bf16": 3375,
    },
    {
        "name": "128 p6-b300",
        "gpus": 1024,
        "nodes": 128,
        "mem_gb": 268,
        "label": "1024_B300",
        "inter_node_bw_gb": 800,
        "intra_node_bw_gbps": 1800,
        "peak_tflops_bf16": 3375,
    },
    {
        "name": "256 p6-b300",
        "gpus": 2048,
        "nodes": 256,
        "mem_gb": 268,
        "label": "2048_B300",
        "inter_node_bw_gb": 800,
        "intra_node_bw_gbps": 1800,
        "peak_tflops_bf16": 3375,
    },
    {
        "name": "512 p6-b300",
        "gpus": 4096,
        "nodes": 512,
        "mem_gb": 268,
        "label": "4096_B300",
        "inter_node_bw_gb": 800,
        "intra_node_bw_gbps": 1800,
        "peak_tflops_bf16": 3375,
    },
]

# ============================================================================
# p6e-gb200.36xlarge (GB200 NVL, 185 GB per GPU, 4 GPUs per node)
# Note: Available as UltraServers only (36 or 72 GPU configs)
# ============================================================================

P6E_GB200_HARDWARE = [
    {
        "name": "9 p6e-gb200 (36 GPUs)",
        "gpus": 36,
        "nodes": 9,
        "mem_gb": 185,
        "label": "36_GB200",
        "inter_node_bw_gb": 400,
        "intra_node_bw_gbps": 900,
        "peak_tflops_bf16": 2500,
    },
    {
        "name": "18 p6e-gb200 (72 GPUs)",
        "gpus": 72,
        "nodes": 18,
        "mem_gb": 185,
        "label": "72_GB200",
        "inter_node_bw_gb": 400,
        "intra_node_bw_gbps": 900,
        "peak_tflops_bf16": 2500,
    },
    {
        "name": "36 p6e-gb200 (144 GPUs)",
        "gpus": 144,
        "nodes": 36,
        "mem_gb": 185,
        "label": "144_GB200",
        "inter_node_bw_gb": 400,
        "intra_node_bw_gbps": 900,
        "peak_tflops_bf16": 2500,
    },
    {
        "name": "72 p6e-gb200 (288 GPUs)",
        "gpus": 288,
        "nodes": 72,
        "mem_gb": 185,
        "label": "288_GB200",
        "inter_node_bw_gb": 400,
        "intra_node_bw_gbps": 900,
        "peak_tflops_bf16": 2500,
    },
]

# ============================================================================
# g5.48xlarge (A10G, 24 GB, 8 GPUs/node)
# Note: Limited NVLink (pairs only), ENA networking, best for inference
# ============================================================================

G5_HARDWARE = [
    {
        "name": "1 g5",
        "gpus": 8,
        "nodes": 1,
        "mem_gb": 24,
        "label": "8_A10G",
        "inter_node_bw_gb": 100,
        "intra_node_bw_gbps": 600,
        "peak_tflops_bf16": 35,
    },
    {
        "name": "4 g5",
        "gpus": 32,
        "nodes": 4,
        "mem_gb": 24,
        "label": "32_A10G",
        "inter_node_bw_gb": 100,
        "intra_node_bw_gbps": 600,
        "peak_tflops_bf16": 35,
    },
    {
        "name": "8 g5",
        "gpus": 64,
        "nodes": 8,
        "mem_gb": 24,
        "label": "64_A10G",
        "inter_node_bw_gb": 100,
        "intra_node_bw_gbps": 600,
        "peak_tflops_bf16": 35,
    },
    {
        "name": "16 g5",
        "gpus": 128,
        "nodes": 16,
        "mem_gb": 24,
        "label": "128_A10G",
        "inter_node_bw_gb": 100,
        "intra_node_bw_gbps": 600,
        "peak_tflops_bf16": 35,
    },
]

# ============================================================================
# g6.48xlarge (L4, 24 GB, 8 GPUs/node)
# Note: PCIe-connected (no NVLink), ENA networking, best for inference
# ============================================================================

G6_HARDWARE = [
    {
        "name": "1 g6",
        "gpus": 8,
        "nodes": 1,
        "mem_gb": 24,
        "label": "8_L4",
        "inter_node_bw_gb": 100,
        "intra_node_bw_gbps": 64,
        "peak_tflops_bf16": 121,
    },
    {
        "name": "4 g6",
        "gpus": 32,
        "nodes": 4,
        "mem_gb": 24,
        "label": "32_L4",
        "inter_node_bw_gb": 100,
        "intra_node_bw_gbps": 64,
        "peak_tflops_bf16": 121,
    },
    {
        "name": "8 g6",
        "gpus": 64,
        "nodes": 8,
        "mem_gb": 24,
        "label": "64_L4",
        "inter_node_bw_gb": 100,
        "intra_node_bw_gbps": 64,
        "peak_tflops_bf16": 121,
    },
    {
        "name": "16 g6",
        "gpus": 128,
        "nodes": 16,
        "mem_gb": 24,
        "label": "128_L4",
        "inter_node_bw_gb": 100,
        "intra_node_bw_gbps": 64,
        "peak_tflops_bf16": 121,
    },
]

# ============================================================================
# g6e.48xlarge (L40S, 48 GB, 8 GPUs/node)
# Note: PCIe-connected (no NVLink), ENA networking, best for inference/fine-tuning
# ============================================================================

G6E_HARDWARE = [
    {
        "name": "1 g6e",
        "gpus": 8,
        "nodes": 1,
        "mem_gb": 48,
        "label": "8_L40S",
        "inter_node_bw_gb": 50,
        "intra_node_bw_gbps": 64,
        "peak_tflops_bf16": 366,
    },
    {
        "name": "4 g6e",
        "gpus": 32,
        "nodes": 4,
        "mem_gb": 48,
        "label": "32_L40S",
        "inter_node_bw_gb": 50,
        "intra_node_bw_gbps": 64,
        "peak_tflops_bf16": 366,
    },
    {
        "name": "8 g6e",
        "gpus": 64,
        "nodes": 8,
        "mem_gb": 48,
        "label": "64_L40S",
        "inter_node_bw_gb": 50,
        "intra_node_bw_gbps": 64,
        "peak_tflops_bf16": 366,
    },
    {
        "name": "16 g6e",
        "gpus": 128,
        "nodes": 16,
        "mem_gb": 48,
        "label": "128_L40S",
        "inter_node_bw_gb": 50,
        "intra_node_bw_gbps": 64,
        "peak_tflops_bf16": 366,
    },
]

# ============================================================================
# Combined list -- include ONLY the platforms the user selected
# ============================================================================

HARDWARE = (
    P5_HARDWARE
    + P5EN_HARDWARE
    + P6_B200_HARDWARE
    + P6_B300_HARDWARE
    + P6E_GB200_HARDWARE
    + G5_HARDWARE
    + G6_HARDWARE
    + G6E_HARDWARE
)
```

**Usage:** Before running the calculator, edit this file to include only the selected platform(s). For example, if the user only wants 128 p6-b200 nodes:

```python
HARDWARE = [
    {
        "name": "128 p6-b200",
        "gpus": 1024,
        "nodes": 128,
        "mem_gb": 180,
        "label": "1024_B200",
        "inter_node_bw_gb": 400,
        "intra_node_bw_gbps": 1800,
        "peak_tflops_bf16": 2250,
    },
]
```
