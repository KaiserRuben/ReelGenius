from celery import Celery
import os
import json
from typing import Dict, Any, Optional
import time
import traceback
from pymongo import MongoClient
from datetime import datetime
import uuid

from .config import PipelineConfig, PlatformType
from .pipeline.pipeline import VideoPipeline

# Configure MongoDB
MONGODB_URL = os.environ.get('MONGODB_URL', 'mongodb://localhost:27017/videogen')
mongo_client = MongoClient(MONGODB_URL)
db = mongo_client['videogen']
tasks_collection = db['tasks']

# Configure Celery
redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
celery_app = Celery('video_generator', broker=redis_url, backend=redis_url)

# Configure task serialization
celery_app.conf.task_serializer = 'json'
celery_app.conf.result_serializer = 'json'
celery_app.conf.accept_content = ['json']
celery_app.conf.result_expires = 60 * 60 * 24  # 24 hours

# Configure concurrency (adjust based on server resources)
celery_app.conf.worker_concurrency = int(os.environ.get('CELERY_WORKERS', '2'))
celery_app.conf.worker_prefetch_multiplier = 1  # Process one task at a time


# Progress reporting
def update_task_status(task_id: str, updates: Dict[str, Any]):
    """Update task status in MongoDB."""
    updates['updated_at'] = datetime.utcnow()
    tasks_collection.update_one(
        {'task_id': task_id},
        {'$set': updates},
        upsert=True
    )


@celery_app.task(bind=True)
def generate_video(self, task_id: str, content: str, platform: str,
                   config_overrides: Optional[Dict[str, Any]] = None):
    """Celery task to generate video."""
    try:
        # Update progress in MongoDB
        update_task_status(task_id, {
            'progress': 0.1,
            'status': 'running',
            'celery_task_id': self.request.id
        })

        # Update config
        task_config = PipelineConfig(platform=platform)

        if config_overrides:
            # Apply config overrides
            config_dict = task_config.model_dump()
            for key, value in config_overrides.items():
                if key in config_dict:
                    if isinstance(value, dict) and isinstance(config_dict[key], dict):
                        config_dict[key].update(value)
                    else:
                        config_dict[key] = value
            task_config = PipelineConfig(**config_dict)

        # Validate API keys
        if not task_config.validate_api_keys():
            update_task_status(task_id, {
                'status': 'failed',
                'error': 'Missing required API keys',
                'progress': 1.0
            })
            return {
                'status': 'failed',
                'error': 'Missing required API keys',
                'progress': 1.0
            }

        # Initialize pipeline
        task_pipeline = VideoPipeline(task_config)

        # Update progress
        update_task_status(task_id, {
            'progress': 0.2,
            'status': 'running'
        })
        self.update_state(state='PROGRESS', meta={
            'progress': 0.2,
            'status': 'running'
        })

        # Run pipeline with progress reporting
        start_time = time.time()

        # Define progress callback
        def progress_callback(progress_value):
            # Calculate overall progress (20% to 90%)
            overall_progress = 0.2 + (progress_value * 0.7)
            update_task_status(task_id, {
                'progress': overall_progress,
                'status': 'running'
            })
            self.update_state(state='PROGRESS', meta={
                'progress': overall_progress,
                'status': 'running'
            })

        # Run pipeline with progress callback
        result = task_pipeline.run(content, progress_callback=progress_callback)

        execution_time = time.time() - start_time

        # Add execution time and platform to result
        result['execution_time'] = execution_time
        result['platform'] = platform

        # Update task status
        final_status = {
            'status': 'completed' if result.get('success', False) else 'failed',
            'progress': 1.0,
            'result': result,
            'platform': platform,
            'execution_time': execution_time
        }

        if not result.get('success', False):
            final_status['error'] = result.get('error')

        update_task_status(task_id, final_status)

        # Return final result
        return final_status

    except Exception as e:
        # Log error
        print(f"Task {task_id} failed: {e}")
        traceback.print_exc()

        # Update status in MongoDB
        update_task_status(task_id, {
            'status': 'failed',
            'error': str(e),
            'progress': 1.0
        })

        # Return error
        return {
            'status': 'failed',
            'error': str(e),
            'progress': 1.0
        }