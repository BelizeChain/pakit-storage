# Pakit Development Dockerfile
FROM python:3.11-slim

LABEL maintainer="BelizeChain Team <development@belizechain.bz>"
LABEL version="1.0.0"
LABEL description="Pakit - Decentralized Storage for BelizeChain"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY pakit_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r pakit_requirements.txt

# Copy application code
COPY . .


# Create logs directory
RUN mkdir -p logs

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run API server
CMD ["python", "api_server.py", "--host", "0.0.0.0", "--port", "8000"]
