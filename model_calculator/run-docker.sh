#!/bin/bash

# LLM Training Calculator - Docker Run Script
# This script builds and runs the LLM training calculator in a Docker container

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="llm-training-calculator"
IMAGE_TAG="latest"
RESULTS_DIR="./results"

echo -e "${BLUE}=== LLM Training Calculator - Docker Runner ===${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Error: Docker is not installed or not in PATH${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo -e "${YELLOW}Error: Docker daemon is not running${NC}"
    echo "Please start Docker Desktop or the Docker daemon"
    exit 1
fi

# Build the image
echo -e "\n${BLUE}Building Docker image...${NC}"
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Docker image built successfully${NC}"
else
    echo -e "${YELLOW}✗ Failed to build Docker image${NC}"
    exit 1
fi

# Create results directory if it doesn't exist
mkdir -p ${RESULTS_DIR}

# Run the container
echo -e "\n${BLUE}Running LLM Training Calculator...${NC}"
echo -e "${BLUE}Output will be saved to: ${RESULTS_DIR}/analysis_output.txt${NC}\n"

docker run --rm \
    -v "$(pwd)/${RESULTS_DIR}:/app/results" \
    ${IMAGE_NAME}:${IMAGE_TAG} \
    | tee ${RESULTS_DIR}/analysis_output.txt

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ Analysis completed successfully${NC}"
    echo -e "${GREEN}Results saved to: ${RESULTS_DIR}/analysis_output.txt${NC}"
else
    echo -e "\n${YELLOW}✗ Analysis failed${NC}"
    exit 1
fi

# Display image info
echo -e "\n${BLUE}Docker Image Information:${NC}"
docker images ${IMAGE_NAME}:${IMAGE_TAG}

echo -e "\n${GREEN}=== Done! ===${NC}"
echo -e "\nTo run again: docker run --rm ${IMAGE_NAME}:${IMAGE_TAG}"
echo -e "To run interactively: docker run --rm -it ${IMAGE_NAME}:${IMAGE_TAG} /bin/bash"
