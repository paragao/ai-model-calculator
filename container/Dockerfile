# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Copy main script
COPY model_calculations.py /app/

# Copy configuration and utility modules
COPY configuration/ /app/configuration/
COPY utils/ /app/utils/

# Set environment variable to ensure Python output is not buffered
ENV PYTHONUNBUFFERED=1

# Make model_calculations.py executable
RUN chmod +x model_calculations.py

# Default command: run the model calculations
CMD ["python3", "model_calculations.py"]
