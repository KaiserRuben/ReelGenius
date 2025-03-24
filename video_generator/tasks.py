from celery import Celery
import os
import json
from typing import Dict, Any, Optional
import time
import traceback
from datetime import datetime
import uuid

from .config import PipelineConfig, PlatformType
from .pipeline.pipeline import VideoPipeline
from .database import Session, Task

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
    """Update task status in PostgreSQL database."""
    session = Session()
    try:
        updates['updated_at'] = datetime.utcnow()
        
        # Check for JSON serialization issues
        if 'result' in updates:
            try:
                # Test if result can be serialized to JSON
                import json
                json_str = json.dumps(updates['result'])
                # Log size for troubleshooting
                result_size = len(json_str)
                print(f"Task {task_id} result JSON size: {result_size} bytes")
                
                # If result is too large, we might need to trim it
                if result_size > 1000000:  # 1MB limit
                    print(f"WARNING: Task {task_id} result is very large ({result_size} bytes)")
                    # Consider trimming large lists or nested data
            except Exception as json_err:
                print(f"ERROR: JSON serialization issue with task {task_id}: {json_err}")
                # Find problematic fields
                for key, value in updates['result'].items():
                    try:
                        json.dumps({key: value})
                    except:
                        print(f"Field '{key}' in result is not JSON serializable")
                        # Remove or simplify problematic fields
                        if isinstance(value, (list, dict)):
                            print(f"Simplifying problematic field: {key}")
                            updates['result'][key] = str(value)[:1000] + "... [truncated]"
                        else:
                            print(f"Removing problematic field: {key}")
                            updates['result'][key] = str(value)[:1000] + "... [truncated]"
        
        # Get task if it exists
        task = session.query(Task).filter_by(task_id=task_id).first()
        
        if task:
            # Update existing task
            for key, value in updates.items():
                setattr(task, key, value)
        else:
            # Create new task
            updates['task_id'] = task_id
            task = Task(**updates)
            session.add(task)
            
        session.commit()
        print(f"Successfully updated task {task_id} with status: {updates.get('status', 'unknown')}")
    except Exception as e:
        session.rollback()
        print(f"Error updating task status: {e}")
        print(f"Task ID: {task_id}, Update keys: {list(updates.keys())}")
        traceback.print_exc()
    finally:
        session.close()


@celery_app.task(bind=True)
def generate_video(self, task_id: str, content: str, platform: str,
                   config_overrides: Optional[Dict[str, Any]] = None):
    """Celery task to generate video."""
    try:
        # Update progress in database
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

        # Define progress callback with task_id support
        def progress_callback(progress_value, task_id=None):
            # Calculate overall progress (20% to 90%)
            overall_progress = 0.2 + (progress_value * 0.7)
            # Create a copy of the progress data without the function itself
            progress_data = {
                'progress': overall_progress,
                'status': 'running'
            }
            # Use the provided task_id if it's passed, otherwise use the outer task_id
            update_task_id = task_id or self.request.id 
            update_task_status(update_task_id, progress_data)
            self.update_state(state='PROGRESS', meta=progress_data)

        # Run pipeline with progress callback and task_id
        result = task_pipeline.run(content, progress_callback=progress_callback, task_id=task_id)

        execution_time = time.time() - start_time

        # Add execution time and platform to result
        result['execution_time'] = execution_time
        result['platform'] = platform
        
        # Clean up the result to ensure it's serializable
        # Remove progress_callback from result to avoid serialization issues
        if 'progress_callback' in result:
            del result['progress_callback']
            
        # Create a clean copy of the result for database storage
        import copy
        clean_result = copy.deepcopy(result)
        
        # Remove any potentially non-serializable objects
        for key in list(clean_result.keys()):
            if callable(clean_result[key]):
                print(f"Removing non-serializable callable: {key}")
                del clean_result[key]
                
        # Check for other problematic data
        if 'processed_scenes' in clean_result and isinstance(clean_result['processed_scenes'], list):
            print(f"Number of scenes found: {len(clean_result['processed_scenes'])}")
            
        # Update task status
        final_status = {
            'status': 'completed' if result.get('success', False) else 'failed',
            'progress': 1.0,
            'result': clean_result,  # Use the cleaned result
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

        # Update status in database
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