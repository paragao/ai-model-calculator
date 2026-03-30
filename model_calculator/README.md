# LLM Training Calculator

A comprehensive tool for analyzing memory requirements, batch configurations, training time estimates, and communication overhead for large language model training across different hardware platforms.

## Overview

This calculator performs detailed analysis across 5 phases:

1. **Phase 1: Memory Analysis** - Calculate GPU memory requirements with ZeRO-1 and ZeRO-2 optimization strategies
2. **Phase 2: Batch Configuration** - Analyze optimal batch sizes and gradient accumulation steps
3. **Phase 3: Training Time Estimation** - Estimate training duration using FLOPs calculations
4. **Phase 4: ZeRO-2 Communication** - Analyze reduce-scatter communication overhead for data parallelism
5. **Phase 4.1: ZeRO-1 Communication** - Analyze all-gather communication overhead for optimizer sharding
6. **Phase 5: All-to-All Communication** - Analyze intra-node MoE routing communication volumes

## Quick Start

### Using Docker (Recommended)

The easiest way to run the calculator is using Docker:

```bash
# Build the Docker image
docker build -t llm-training-calculator:latest .

# Run the calculator
docker run --rm llm-training-calculator:latest

# Or use the convenience script
./run-docker.sh
```

**See [README-DOCKER.md](README-DOCKER.md) for detailed Docker usage instructions.**

### Running Directly with Python

Requirements: Python 3.8+

```bash
# Run the calculator
python3 model_calculations.py
```

No external dependencies required - uses only Python standard library!

## Features

✅ **Memory Optimization** - Automatic selection between ZeRO-1 and ZeRO-2 based on memory constraints
✅ **Multiple Hardware Platforms** - Analyze across H100, H200, and custom hardware configurations
✅ **Model Variants** - Compare different model architectures and parameter counts
✅ **Batch Size Optimization** - Find optimal micro-batch and accumulation settings
✅ **Training Time Estimates** - Calculate training duration with MFU considerations
✅ **Communication Analysis** - Identify bottlenecks in distributed training
✅ **Color-Coded Output** - Easy-to-read results with performance indicators
✅ **Export Options** - Save results to CSV/JSON for further analysis

## Architecture

The codebase is organized into modular components:

```
model_calculations.py          # Main orchestration script (572 lines)
├── validation.py               # Configuration validation (67 lines)
├── core_calculations.py        # Shared calculation functions (280 lines)
├── formatting_utils.py         # Color output and formatting (180 lines)
├── phase1_memory.py           # Phase 1: Memory Analysis (383 lines)
├── phase2_batch.py            # Phase 2: Batch Configuration (164 lines)
├── phase3_training.py         # Phase 3: Training Time (337 lines)
├── phase4_zero2_comm.py       # Phase 4: ZeRO-2 Communication (241 lines)
├── phase4_1_zero1_comm.py     # Phase 4.1: ZeRO-1 Communication (241 lines)
└── phase5_alltoall_comm.py    # Phase 5: All-to-All Communication (230 lines)

Configuration files:
├── variants_config.py         # Model variant definitions
├── hardware_config.py         # Hardware platform specifications
├── project_config.py          # Training parameters
└── advanced_config.py         # Parallelization settings (PP, TP, EP, CP)
```

**Total: 2,123 lines of clean, modular Python code**

## Configuration

### Define Model Variants

Edit `variants_config.py`:

```python
VARIANTS = [
    {
        "name": "A",
        "d": 12288,              # Hidden dimension
        "layers": 64,            # Number of layers
        "total_params_B": 671,   # Total parameters (billions)
        "active_params_B": 53,   # Active parameters per forward pass
    },
    # Add more variants...
]
```

### Configure Hardware Platforms

Edit `hardware_config.py`:

```python
HARDWARE = [
    {
        "name": "4,096 H200",
        "nodes": 512,
        "gpus": 4096,
        "mem_gb": 141,
        "peak_tflops_bf16": 1979,
        "inter_node_bw_gb": 200,
        "intra_node_bw_gbps": 900,
    },
    # Add more hardware configs...
]
```

### Set Training Parameters

Edit `project_config.py`:

```python
TOTAL_TOKENS = 15e12           # Total training tokens (15T)
TOKENS_PER_BATCH = 4e6         # Target tokens per batch (4M)
SEQ_LEN = 8192                 # Sequence length
MICROS = [1, 2, 4]            # Micro batch sizes to test
```

### Configure Parallelization

Edit `advanced_config.py`:

```python
PP = 1          # Pipeline Parallelism
TP = 1          # Tensor Parallelism
EP = 8          # Expert Parallelism
CP = 1          # Context Parallelism
N_EXPERTS = 128 # Total number of experts
```

## Output Examples

### Phase 1: Memory Analysis

```
  Variant       Model     Grad    Optim    Activ      Buf    TOTAL  Headroom  Util%   ZeRO  micro
  ---------- -------- -------- -------- -------- -------- -------- --------- ------ ------ ------
  A             46.4G    46.4G     0.1G    18.1G     5.1G   116.1G    24.9G    82%     Z1      4
  B             49.2G    49.2G     0.1G    18.5G     5.6G   122.6G    18.4G    87%     Z1      4
```

### Phase 3: Training Time Estimates

```
  Platform                  GPUs  ZeRO  micro  accum  Tok/batch  Steps  Rel Speed  Est. Time
  ------------------------- ----- ----- ------ ------ ---------- ------ ---------- -----------
  4,096 H200 (4.0M)          4096    Z1      4     61      4.0M  3,750       1.0x    1.5-1.7mo
  4,096 H100 (4.0M)          4096    Z2      4     61      4.0M  3,750       0.6x    2.4-2.8mo
```

## Advanced Features

### Export Results

Uncomment export functions in `model_calculations.py` to save results:

```python
# At the end of each phase, add:
export_results_csv(all_results, "phase1_memory_results.csv")
export_results_json(all_results, "phase1_memory_results.json")
```

### Custom Analysis

Import and use individual phase modules:

```python
from phase1_memory import analyze_variant_memory
from phase3_training import calculate_training_metrics

# Run custom analysis...
```

## Distribution

Want to share this tool with others?

📦 **See [DISTRIBUTION-GUIDE.md](DISTRIBUTION-GUIDE.md) for comprehensive instructions on:**
- Sharing via Docker Hub
- Creating tarball distributions
- Sharing source code packages
- Git repository setup
- Customization options for recipients

## System Requirements

### For Docker
- Docker Engine 20.10+ or Docker Desktop
- 512MB RAM minimum
- No GPU required

### For Python
- Python 3.8 or higher
- No external dependencies (uses standard library only)

## Troubleshooting

### Colors not displaying
Disable ANSI colors:
```bash
# For Python
export USE_COLOR=False
python3 model_calculations.py

# For Docker
docker run --rm -e USE_COLOR=False llm-training-calculator:latest
```

### Out of Memory (OOM) warnings
This is expected! The tool analyzes which configurations fit in GPU memory. OOM warnings indicate that a particular variant doesn't fit on that hardware configuration.

### Import errors
Ensure all Python files are in the same directory:
```bash
ls *.py
# Should show all 14 Python files
```

## Contributing

Contributions are welcome! Areas for improvement:
- Additional hardware platform configurations
- More sophisticated communication models
- Pipeline parallelism analysis
- Activation checkpointing support
- Extended MoE routing strategies

## Performance

The calculator is computationally lightweight:
- **Execution time**: ~1-5 seconds for standard configurations
- **Memory usage**: <100MB RAM
- **Container size**: 212MB (Docker image)

## Acknowledgments

Built using:
- Python 3.11 standard library
- Docker for containerization
- Mathematical models based on:
  - ZeRO optimization (Microsoft DeepSpeed)
  - Transformer architecture analysis
  - FLOPs-based training time estimation
  - Communication overhead modeling

## License

[Add your license here - MIT, Apache 2.0, GPL, etc.]

## Contact

[Add your contact information here]

---

**Happy calculating! 🚀**

For detailed Docker usage, see [README-DOCKER.md](README-DOCKER.md)
For distribution instructions, see [DISTRIBUTION-GUIDE.md](DISTRIBUTION-GUIDE.md)
