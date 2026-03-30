# LLM Training Calculator - Distribution Guide

This guide explains how to package, share, and distribute the LLM Training Calculator Docker container.

## 📦 What's Included

The Docker container packages:
- **15 Python modules** (model_calculations.py + 14 supporting modules)
- **Zero external dependencies** (uses only Python 3.11 standard library)
- **212MB compressed image** size
- **All 5 analysis phases** (Memory, Batch, Training Time, Communication Analysis)

## 🚀 Quick Distribution Methods

### Method 1: Share via Docker Hub (Recommended)

**For the distributor:**

```bash
# 1. Tag your image
docker tag llm-training-calculator:latest yourusername/llm-training-calculator:latest

# 2. Login to Docker Hub
docker login

# 3. Push to Docker Hub
docker push yourusername/llm-training-calculator:latest
```

**For the recipient:**

```bash
# Pull and run in one command
docker run --rm yourusername/llm-training-calculator:latest

# Or pull first, then run
docker pull yourusername/llm-training-calculator:latest
docker run --rm llm-training-calculator:latest
```

---

### Method 2: Share as Tarball File

**For the distributor:**

```bash
# Save image to compressed tarball (~212MB)
docker save llm-training-calculator:latest | gzip > llm-calculator.tar.gz

# Share llm-calculator.tar.gz via:
# - Email (if size permits)
# - Cloud storage (Google Drive, Dropbox, S3)
# - File transfer service (WeTransfer, etc.)
# - USB drive
```

**For the recipient:**

```bash
# Load image from tarball
gunzip -c llm-calculator.tar.gz | docker load

# Run the calculator
docker run --rm llm-training-calculator:latest
```

---

### Method 3: Share Source Code + Dockerfile

**For the distributor:**

```bash
# Create a distribution package
cd /path/to/project
tar -czf llm-calculator-source.tar.gz \
  *.py \
  Dockerfile \
  docker-compose.yml \
  run-docker.sh \
  .dockerignore \
  README-DOCKER.md

# Share llm-calculator-source.tar.gz
```

**For the recipient:**

```bash
# Extract the package
tar -xzf llm-calculator-source.tar.gz
cd llm-calculator-source

# Build and run using the convenience script
./run-docker.sh

# Or build manually
docker build -t llm-training-calculator:latest .
docker run --rm llm-training-calculator:latest
```

---

### Method 4: Share via Git Repository

**For the distributor:**

```bash
# Initialize git repository (if not already done)
git init
git add *.py Dockerfile docker-compose.yml run-docker.sh .dockerignore README-DOCKER.md
git commit -m "Initial commit: LLM Training Calculator"

# Push to GitHub/GitLab/Bitbucket
git remote add origin https://github.com/yourusername/llm-training-calculator.git
git push -u origin main
```

**For the recipient:**

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-training-calculator.git
cd llm-training-calculator

# Build and run
./run-docker.sh
```

---

## 📋 Files to Share

### Essential Files (Minimum)
```
model_calculations.py          # Main orchestration script
validation.py                  # Configuration validation
core_calculations.py           # Shared calculation functions
formatting_utils.py            # Output formatting
phase1_memory.py              # Phase 1: Memory Analysis
phase2_batch.py               # Phase 2: Batch Configuration
phase3_training.py            # Phase 3: Training Time
phase4_zero2_comm.py          # Phase 4: ZeRO-2 Communication
phase4_1_zero1_comm.py        # Phase 4.1: ZeRO-1 Communication
phase5_alltoall_comm.py       # Phase 5: All-to-All Communication
variants_config.py            # Model variant definitions
hardware_config.py            # Hardware platform specs
project_config.py             # Training parameters
advanced_config.py            # Parallelization settings
Dockerfile                    # Docker build instructions
```

### Recommended Additional Files
```
docker-compose.yml            # Easy container management
run-docker.sh                 # Automated build & run script
.dockerignore                 # Docker build optimization
README-DOCKER.md              # User documentation
DISTRIBUTION-GUIDE.md         # This file
```

---

## 🎯 Distribution Options Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Docker Hub** | Easy to use, automatic updates, version control | Requires Docker Hub account, public by default | Wide distribution, frequent updates |
| **Tarball** | No accounts needed, offline transfer, single file | Large file size (~212MB), no version control | One-time sharing, offline environments |
| **Source Code** | Most flexible, allows customization, smallest transfer | Requires building, longer setup time | Developers, customization needed |
| **Git Repository** | Version control, collaboration, documentation | Requires git knowledge, build step needed | Active development, team collaboration |

---

## 🔧 Customization for Recipients

Recipients can customize the calculator by:

### 1. Using Custom Configuration Files

```bash
# Create custom configuration
cat > custom_variants.py <<EOF
VARIANTS = [
    {
        "name": "MyModel",
        "d": 8192,
        "layers": 64,
        # ... more config
    }
]
EOF

# Run with custom config
docker run --rm \
  -v $(pwd)/custom_variants.py:/app/variants_config.py \
  llm-training-calculator:latest
```

### 2. Using Docker Compose with Overrides

```bash
# Edit docker-compose.yml to uncomment volume mounts
docker-compose up
```

### 3. Rebuilding with Modified Files

```bash
# Edit Python files
vim variants_config.py

# Rebuild image
docker build -t llm-training-calculator:latest .

# Run updated version
docker run --rm llm-training-calculator:latest
```

---

## 💾 Saving Output

### Save to File
```bash
docker run --rm llm-training-calculator:latest > analysis_results.txt
```

### Save with Timestamp
```bash
docker run --rm llm-training-calculator:latest > analysis_$(date +%Y%m%d_%H%M%S).txt
```

### Mount Volume for CSV/JSON Exports
```bash
mkdir results
docker run --rm -v $(pwd)/results:/app/results llm-training-calculator:latest
```

---

## 🐛 Troubleshooting for Recipients

### Container won't start
```bash
# Check if Docker is running
docker info

# Verify image exists
docker images llm-training-calculator

# Check Docker logs
docker logs <container-id>
```

### Colors not displaying
```bash
# Disable colors
docker run --rm -e USE_COLOR=False llm-training-calculator:latest
```

### Permission issues with mounted volumes
```bash
# Fix permissions (Linux/Mac)
chmod -R 755 ./results
```

---

## 📊 Image Information

```bash
# View image details
docker images llm-training-calculator:latest

# Inspect image layers
docker history llm-training-calculator:latest

# View container size
docker ps -s
```

---

## 🔒 Security Considerations

- **No sensitive data** is included in the container
- Container runs with **default user privileges** (not root)
- **No network access** required (purely computational)
- **Read-only filesystems** recommended for production use:
  ```bash
  docker run --rm --read-only llm-training-calculator:latest
  ```

---

## 📝 License and Attribution

When sharing, consider including:
- License file (MIT, Apache, GPL, etc.)
- Attribution requirements
- Usage restrictions (if any)
- Contact information for support

---

## 🆘 Support for Recipients

Recipients should have:
- **Docker installed** (20.10+ or Docker Desktop)
- **512MB RAM minimum** (for container)
- **No GPU required** (calculations only)
- **Basic Docker knowledge** (optional but helpful)

### Getting Help

Include these support resources:
1. README-DOCKER.md - Basic usage instructions
2. This distribution guide - Detailed sharing instructions
3. Contact email or issue tracker
4. Example configurations

---

## ✅ Pre-Distribution Checklist

Before sharing, verify:

- [ ] Docker image builds successfully
- [ ] Container runs without errors
- [ ] Output is correct and complete
- [ ] Documentation is included (README-DOCKER.md)
- [ ] License file is present (if applicable)
- [ ] Example configurations are provided
- [ ] Version/date is documented
- [ ] Contact information is included

---

## 📦 Example Distribution Package Structure

```
llm-training-calculator/
├── README.md                  # Overview and quick start
├── README-DOCKER.md          # Docker usage guide
├── DISTRIBUTION-GUIDE.md     # This file
├── LICENSE                   # License file
├── Dockerfile                # Docker build file
├── docker-compose.yml        # Docker Compose config
├── run-docker.sh            # Convenience script
├── .dockerignore            # Docker ignore file
├── model_calculations.py    # Main script
├── validation.py            # Validation module
├── core_calculations.py     # Core functions
├── formatting_utils.py      # Formatting utilities
├── phase1_memory.py         # Phase 1 module
├── phase2_batch.py          # Phase 2 module
├── phase3_training.py       # Phase 3 module
├── phase4_zero2_comm.py     # Phase 4 module
├── phase4_1_zero1_comm.py   # Phase 4.1 module
├── phase5_alltoall_comm.py  # Phase 5 module
├── variants_config.py       # Model configurations
├── hardware_config.py       # Hardware specifications
├── project_config.py        # Project settings
├── advanced_config.py       # Advanced settings
└── examples/                # Example configurations
    ├── custom_variants.py
    ├── custom_hardware.py
    └── README.md
```

---

## 🚀 Quick Command Reference

```bash
# Build
docker build -t llm-training-calculator:latest .

# Run
docker run --rm llm-training-calculator:latest

# Run with output saved
docker run --rm llm-training-calculator:latest > results.txt

# Run with custom config
docker run --rm -v $(pwd)/custom.py:/app/variants_config.py llm-training-calculator:latest

# Run interactively
docker run --rm -it llm-training-calculator:latest /bin/bash

# Save to tarball
docker save llm-training-calculator:latest | gzip > llm-calculator.tar.gz

# Load from tarball
gunzip -c llm-calculator.tar.gz | docker load

# Push to Docker Hub
docker push yourusername/llm-training-calculator:latest
```

---

**Ready to distribute! 🎉**

Choose your preferred distribution method above and share your LLM Training Calculator with the world!
