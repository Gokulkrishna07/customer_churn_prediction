# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY api/requirements.txt /app/api/

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/api/requirements.txt

# Copy training code and models
COPY training/ /app/training/

# Copy compatibility modules for pickle
COPY preprocessing.py /app/
COPY data_loader.py /app/

# Copy API code
COPY api/ /app/api/

# Set Python path to find training modules
ENV PYTHONPATH=/app:${PYTHONPATH}

# Change to API directory
WORKDIR /app/api

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "main.py"]