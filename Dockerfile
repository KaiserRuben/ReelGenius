FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
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

# Run the application
CMD ["uvicorn", "video_generator.api:app", "--host", "0.0.0.0", "--port", "8000"]


