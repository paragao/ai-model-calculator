# Container and Distribution Files

This directory contains all Docker containerization and distribution-related files for the LLM Training Calculator.

## Files in This Directory

### Docker Files
- **Dockerfile** - Container build configuration
- **docker-compose.yml** - Docker Compose configuration for multi-container setup
- **run-docker.sh** - Convenience script to build and run the container

### Documentation
- **README-DOCKER.md** - Comprehensive Docker usage guide
- **CONTAINER-SUMMARY.txt** - Container build and deployment summary
- **DISTRIBUTION-GUIDE.md** - Guide for distributing the tool (Docker Hub, tarball, git)

## Quick Start

### Build and Run with Script

From the **parent directory**:
```bash
./container/run-docker.sh
```

Or from **this directory**:
```bash
./run-docker.sh
```

The script will automatically:
1. Build the Docker image
2. Run the calculator
3. Save results to `results/analysis_output.txt`

### Build Manually

From the **parent directory** (model_calculator/):
```bash
# Build
docker build -f container/Dockerfile -t llm-training-calculator:latest .

# Run
docker run --rm llm-training-calculator:latest
```

### Using Docker Compose

From the **parent directory**:
```bash
docker-compose -f container/docker-compose.yml up
```

## Why a Separate Container Directory?

Organizing container-related files in a dedicated directory:
- ✅ Keeps the root directory clean
- ✅ Makes containerization optional and clearly separated
- ✅ Groups all distribution-related documentation together
- ✅ Easier to find Docker/deployment files

## Distribution Options

See [DISTRIBUTION-GUIDE.md](DISTRIBUTION-GUIDE.md) for comprehensive instructions on:
- Sharing via Docker Hub
- Creating tarball distributions
- Sharing source code packages
- Git repository setup

## For More Information

- Docker usage details: [README-DOCKER.md](README-DOCKER.md)
- Main project README: [../README.md](../README.md)
