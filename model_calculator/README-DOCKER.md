# LLM Training Calculator - Docker Container

This Docker container packages the LLM Training Calculator tool, which analyzes memory requirements, batch configurations, training time estimates, and communication overhead for large language model training across different hardware platforms.

## Features

The calculator performs 5 comprehensive analysis phases:
1. **Memory Analysis** - Calculate memory requirements per GPU with ZeRO optimization strategies
2. **Batch Configuration** - Analyze optimal batch sizes and gradient accumulation steps
3. **Training Time Estimation** - Estimate training duration across different hardware platforms
4. **ZeRO-2/ZeRO-1 Communication** - Analyze inter-node communication overhead for data parallelism
5. **All-to-All Communication** - Analyze intra-node MoE routing communication volumes

## Quick Start

### Build the Docker Image

```bash
docker build -t llm-training-calculator .
```

### Run the Calculator

```bash
docker run --rm llm-training-calculator
```

The tool will output comprehensive analysis tables with color-coded results to the console.

## Configuration

### Modifying Model Variants

To analyze different model configurations, you need to modify the configuration files before building the image:

1. **variants_config.py** - Define model architectures (layers, dimensions, parameters)
2. **hardware_config.py** - Define hardware platforms (GPUs, memory, bandwidth)
3. **project_config.py** - Set training parameters (total tokens, sequence length, batch size)
4. **advanced_config.py** - Configure parallelization strategies (PP, TP, EP, CP)

After making changes, rebuild the Docker image:

```bash
docker build -t llm-training-calculator .
```

### Running with Custom Configurations

You can mount custom configuration files at runtime:

```bash
docker run --rm \
  -v $(pwd)/custom_variants.py:/app/variants_config.py \
  -v $(pwd)/custom_hardware.py:/app/hardware_config.py \
  llm-training-calculator
```

### Saving Output to Files

To save the analysis output to a file on your host machine:

```bash
docker run --rm llm-training-calculator > analysis_results.txt
```

Or mount a volume to save CSV/JSON exports:

```bash
docker run --rm -v $(pwd)/results:/app/results llm-training-calculator
```

## Advanced Usage

### Interactive Container

To explore the container interactively:

```bash
docker run --rm -it llm-training-calculator /bin/bash
```

Inside the container, you can:
- View configuration files: `cat variants_config.py`
- Run the calculator: `python3 model_calculations.py`
- Modify configs and re-run analysis

### Using Docker Compose (Optional)

Create a `docker-compose.yml` file:

```yaml
version: '3.8'
services:
  calculator:
    build: .
    volumes:
      - ./results:/app/results
      - ./custom_configs:/app/configs:ro
```

Run with:
```bash
docker-compose up
```

## Architecture

The calculator is organized into modular components:

```
model_calculations.py          # Main orchestration script
├── validation.py               # Configuration validation
├── core_calculations.py        # Shared calculation functions
├── formatting_utils.py         # Color output and formatting
├── phase1_memory.py           # Phase 1: Memory Analysis
├── phase2_batch.py            # Phase 2: Batch Configuration
├── phase3_training.py         # Phase 3: Training Time
├── phase4_zero2_comm.py       # Phase 4: ZeRO-2 Communication
├── phase4_1_zero1_comm.py     # Phase 4.1: ZeRO-1 Communication
└── phase5_alltoall_comm.py    # Phase 5: All-to-All Communication

Configuration files:
├── variants_config.py         # Model variant definitions
├── hardware_config.py         # Hardware platform specs
├── project_config.py          # Training parameters
└── advanced_config.py         # Parallelization settings
```

## Output Format

The calculator outputs:
- **Console**: Formatted tables with color-coded metrics and recommendations
- **CSV Export**: Detailed results for each analysis phase (optional)
- **JSON Export**: Structured data for programmatic processing (optional)

## System Requirements

- Docker Engine 20.10+ or Docker Desktop
- 512MB RAM minimum (for container)
- No GPU required (calculations only, no actual training)

## Sharing the Container

### Save Image to File

```bash
docker save llm-training-calculator:latest | gzip > llm-calculator.tar.gz
```

### Load Image from File

```bash
docker load < llm-calculator.tar.gz
```

### Push to Docker Hub (Optional)

```bash
# Tag the image
docker tag llm-training-calculator:latest yourusername/llm-training-calculator:latest

# Push to Docker Hub
docker push yourusername/llm-training-calculator:latest
```

Others can then pull and run:
```bash
docker pull yourusername/llm-training-calculator:latest
docker run --rm yourusername/llm-training-calculator:latest
```

## Troubleshooting

### Colors not displaying correctly
If ANSI color codes aren't rendering, disable colors by setting the environment variable:
```bash
docker run --rm -e USE_COLOR=False llm-training-calculator
```

### Permission issues with mounted volumes
Ensure the mounted directories have appropriate permissions:
```bash
chmod -R 755 ./results
```

### Out of memory in container
Increase Docker's memory limit in Docker Desktop settings or use `--memory` flag:
```bash
docker run --rm --memory=1g llm-training-calculator
```

## License

[Add your license information here]

## Contact

[Add contact information here]
