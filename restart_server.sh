#!/bin/bash
# Script to restart the ReelGenius server with safer settings

# Kill any running Celery workers
echo "Stopping any running Celery workers..."
pkill -f "celery worker"

# Wait for processes to terminate
sleep 2

# Start API with specific settings
echo "Starting API server..."
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="" USE_CUDA=0 uvicorn video_generator.api:app --host 0.0.0.0 --port 8000 &

# Wait for API to start
sleep 5

# Start Celery worker with solo pool to avoid forking issues
echo "Starting Celery worker with safe configuration and semantic cache disabled..."
DISABLE_SEMANTIC_CACHE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="" USE_CUDA=0 celery -A video_generator.tasks worker --loglevel=info -P solo --concurrency=1

echo "Server startup complete!"