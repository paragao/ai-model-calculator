# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Copy entry point and analysis engine
COPY ai_model_calculator.py /app/
COPY model_calculations.py /app/

# Copy configuration and utility modules
COPY configuration/ /app/configuration/
COPY utils/ /app/utils/

# Create results directory for optional export mounts
RUN mkdir -p /app/results

# Set environment variable to ensure Python output is not buffered
ENV PYTHONUNBUFFERED=1

# Make entry points executable
RUN chmod +x ai_model_calculator.py model_calculations.py

# Use ENTRYPOINT so users can pass flags (e.g. --model, --json, --report)
ENTRYPOINT ["python3", "ai_model_calculator.py"]

# Empty CMD — default runs full verbose output; pass --report for concise summary
CMD []
