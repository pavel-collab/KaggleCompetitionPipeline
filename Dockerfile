FROM python:3.11-slim

WORKDIR /app

# System dependencies (required for kaggle)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Default command (overridden in docker-compose)
CMD ["python", "-m", "app.main", "scheduler"]
