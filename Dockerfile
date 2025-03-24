FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/inputs data/history output/videos output/metadata

# Set environment variables
ENV PYTHONPATH=/app

# Set up entrypoint script
RUN chmod +x /app/docker-entrypoint.sh

# Use custom entrypoint script
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command (will be passed to the entrypoint)
CMD ["api"]

