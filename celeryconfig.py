# Celery Configuration File
# This file configures Celery to work with fork-sensitive libraries like SentenceTransformers

# Basic Celery configuration
broker_url = 'redis://localhost:6379/0'  # Use Docker service name 'redis' in production
result_backend = 'redis://localhost:6379/0'

# Task routing
task_routes = {
    'video_generator.tasks.generate_video': {'queue': 'video_generation'}
}

# Task serialization
task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
result_expires = 3600  # Results expire after 1 hour

# Worker settings
worker_prefetch_multiplier = 1  # Reduced to prevent memory bloating
worker_max_tasks_per_child = 10  # Restart worker after 10 tasks to free memory

# Important: Force 'fork' instead of 'spawn' on macOS
# This is critical for proper handling of SentenceTransformers
import platform
if platform.system() == 'Darwin':  # macOS
    worker_pool_restarts = True

# Important: Configure worker pool settings
worker_pool = 'solo'  # Use 'solo' pool to avoid forking issues

# Configure concurrency
worker_concurrency = 1  # Use single process to avoid forking issues

# Set reasonable task time limits
task_time_limit = 3600  # 1 hour max
task_soft_time_limit = 1800  # 30 min soft limit (warning)

# Set reasonable task result size
result_compression = 'gzip'
result_expires = 86400  # Keep results for 1 day

# Optional: Enable task events for monitoring
worker_send_task_events = True
task_send_sent_event = True