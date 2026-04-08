FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Environment variables
ENV SERVICE_NAME=drift-detector
ENV LOG_LEVEL=INFO
ENV WINDOW_SIZE=1000
ENV REFERENCE_WINDOW_SIZE=5000
ENV DRIFT_CHECK_INTERVAL=300
ENV PSI_THRESHOLD=0.2
ENV KS_THRESHOLD=0.05

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose port
EXPOSE 8001

# Run the service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
