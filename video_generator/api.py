from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Query, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Union, Literal
import os
import json
import shutil
from pathlib import Path
import time
import uuid
import asyncio
from loguru import logger
import traceback
from pymongo import MongoClient
from datetime import datetime, timedelta

from .config import PipelineConfig, PlatformType
from .tasks import celery_app, generate_video

app = FastAPI(title="AI Video Generator API")

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize configuration
config = PipelineConfig()

# Configure MongoDB
MONGODB_URL = os.environ.get('MONGODB_URL', 'mongodb://localhost:27017/videogen')
mongo_client = MongoClient(MONGODB_URL)
db = mongo_client['videogen']
tasks_collection = db['tasks']

# Ensure output directories exist
output_dir = Path(__file__).parent.parent / "output"
videos_dir = output_dir / "videos"
metadata_dir = output_dir / "metadata"
temp_dir = output_dir / "temp"
os.makedirs(videos_dir, exist_ok=True)
os.makedirs(metadata_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

# Task cleanup configuration
TASK_RETENTION_PERIOD = int(os.environ.get("TASK_RETENTION_PERIOD", "86400"))  # Default 24 hours


# Models
class VideoGenerationRequest(BaseModel):
    content: str
    platform: Optional[PlatformType] = "tiktok"
    config_overrides: Optional[Dict[str, Any]] = None


class VideoGenerationResponse(BaseModel):
    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Cleanup function
async def cleanup_old_tasks():
    """Clean up old tasks from MongoDB"""
    cutoff_time = datetime.utcnow() - timedelta(seconds=TASK_RETENTION_PERIOD)
    old_tasks = tasks_collection.find({"created_at": {"$lt": cutoff_time}})

    for task in old_tasks:
        logger.info(f"Cleaning up old task: {task['task_id']}")

        try:
            # Delete related files if they exist
            if "result" in task and "video_path" in task["result"]:
                video_path = task["result"]["video_path"]
                if os.path.exists(video_path):
                    os.remove(video_path)
                    logger.info(f"Deleted video file: {video_path}")

            # Delete metadata if it exists
            if "result" in task and "metadata" in task["result"] and "metadata_path" in task["result"]["metadata"]:
                metadata_path = task["result"]["metadata"]["metadata_path"]
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                    logger.info(f"Deleted metadata file: {metadata_path}")
        except Exception as e:
            logger.error(f"Error cleaning up files for task {task['task_id']}: {e}")

        # Remove from MongoDB
        tasks_collection.delete_one({"task_id": task["task_id"]})


# Schedule periodic cleanup
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(periodic_cleanup())
    logger.info("API server started")


async def periodic_cleanup():
    """Run cleanup periodically"""
    cleanup_interval = int(os.environ.get("TASK_CLEANUP_INTERVAL", "3600"))  # Default 1 hour
    while True:
        await cleanup_old_tasks()
        await asyncio.sleep(cleanup_interval)


# API Routes
@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "AI Video Generator API", "status": "running", "version": "1.0.0"}


@app.post("/generate", response_model=VideoGenerationResponse)
async def create_video(request: VideoGenerationRequest):
    """Generate a video from content."""
    try:
        # Validate input
        if not request.content or len(request.content.strip()) < 10:
            raise HTTPException(status_code=400, detail="Content must be at least 10 characters long")

        # Generate task ID
        task_id = str(uuid.uuid4())

        # Store initial task info in MongoDB
        tasks_collection.insert_one({
            "task_id": task_id,
            "status": "queued",
            "progress": 0,
            "created_at": datetime.utcnow(),
            "platform": request.platform,
            "content_summary": request.content[:100] + "..." if len(request.content) > 100 else request.content
        })

        # Start Celery task
        celery_result = generate_video.apply_async(
            args=[task_id, request.content, request.platform, request.config_overrides],
            task_id=task_id
        )

        # Update MongoDB with Celery task ID
        tasks_collection.update_one(
            {"task_id": task_id},
            {"$set": {"celery_task_id": celery_result.id}}
        )

        logger.info(f"Video generation task queued: {task_id}")
        return VideoGenerationResponse(
            task_id=task_id,
            status="queued",
            message="Video generation task has been queued"
        )
    except Exception as e:
        logger.error(f"Error in generate_video: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/generate-from-file", response_model=VideoGenerationResponse)
async def generate_video_from_file(
        file: UploadFile = File(...),
        platform: PlatformType = Form("tiktok"),
        config_overrides: Optional[str] = Form(None)
):
    """Generate a video from an uploaded file."""
    # Save uploaded file temporarily
    temp_file = Path(temp_dir) / f"{uuid.uuid4()}_{file.filename}"
    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Read content from file
        try:
            with open(temp_file, "r") as f:
                content = f.read()

            # Validate content
            if not content or len(content.strip()) < 10:
                raise HTTPException(status_code=400, detail="Content must be at least 10 characters long")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="Uploaded file must be text")

        # Parse config overrides if provided
        overrides = None
        if config_overrides:
            try:
                overrides = json.loads(config_overrides)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in config_overrides")

        # Generate task ID
        task_id = str(uuid.uuid4())

        # Store initial task info in MongoDB
        tasks_collection.insert_one({
            "task_id": task_id,
            "status": "queued",
            "progress": 0,
            "created_at": datetime.utcnow(),
            "platform": platform,
            "content_source": file.filename,
            "content_summary": content[:100] + "..." if len(content) > 100 else content
        })

        # Start Celery task
        celery_result = generate_video.apply_async(
            args=[task_id, content, platform, overrides],
            task_id=task_id
        )

        # Update MongoDB with Celery task ID
        tasks_collection.update_one(
            {"task_id": task_id},
            {"$set": {"celery_task_id": celery_result.id}}
        )

        logger.info(f"Video generation task from file queued: {task_id}")
        return VideoGenerationResponse(
            task_id=task_id,
            status="queued",
            message="Video generation task has been queued"
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in generate_video_from_file: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_file.exists():
            temp_file.unlink()


@app.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get the status of a video generation task."""
    # First check MongoDB
    task_info = tasks_collection.find_one({"task_id": task_id})

    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")

    # If task has a Celery ID and isn't completed yet, check Celery for latest status
    if "celery_task_id" in task_info and task_info["status"] not in ["completed", "failed"]:
        try:
            celery_task = celery_app.AsyncResult(task_info["celery_task_id"])

            if celery_task.state == 'SUCCESS' and celery_task.result:
                # Update from Celery result
                result = celery_task.result
                tasks_collection.update_one(
                    {"task_id": task_id},
                    {"$set": {
                        "status": result.get("status", "completed"),
                        "progress": result.get("progress", 1.0),
                        "result": result.get("result", {}),
                        "error": result.get("error"),
                        "execution_time": result.get("execution_time"),
                        "updated_at": datetime.utcnow()
                    }}
                )
                # Refresh task info
                task_info = tasks_collection.find_one({"task_id": task_id})

            elif celery_task.state == 'FAILURE':
                # Update with failure info
                tasks_collection.update_one(
                    {"task_id": task_id},
                    {"$set": {
                        "status": "failed",
                        "progress": 1.0,
                        "error": str(celery_task.result),
                        "updated_at": datetime.utcnow()
                    }}
                )
                # Refresh task info
                task_info = tasks_collection.find_one({"task_id": task_id})

            # Pending and progress states are already updated by the task itself

        except Exception as e:
            logger.error(f"Error checking Celery task status: {e}")
            # Continue with current MongoDB status

    # Convert MongoDB ObjectId for proper JSON serialization
    if "_id" in task_info:
        del task_info["_id"]

    # Create response with required fields
    return TaskStatusResponse(
        task_id=task_id,
        status=task_info.get("status", "unknown"),
        progress=task_info.get("progress"),
        result=task_info.get("result"),
        error=task_info.get("error")
    )


@app.get("/video/{task_id}")
async def get_video(task_id: str):
    """Get the generated video for a completed task."""
    task_info = tasks_collection.find_one({"task_id": task_id})

    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")

    if task_info["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Task is not completed: {task_info['status']}")

    if "result" not in task_info or "video_path" not in task_info["result"]:
        raise HTTPException(status_code=400, detail="Video path not found in task result")

    video_path = task_info["result"]["video_path"]

    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=os.path.basename(video_path)
    )


@app.get("/metadata/{task_id}")
async def get_metadata(task_id: str):
    """Get the metadata for a completed task."""
    task_info = tasks_collection.find_one({"task_id": task_id})

    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")

    if task_info["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Task is not completed: {task_info['status']}")

    if "result" not in task_info or "metadata" not in task_info["result"]:
        raise HTTPException(status_code=400, detail="Metadata not found in task result")

    return task_info["result"]["metadata"]


@app.get("/tasks")
async def list_tasks(
        status: Optional[str] = Query(None, description="Filter by status"),
        platform: Optional[PlatformType] = Query(None, description="Filter by platform"),
        limit: int = Query(10, description="Maximum number of tasks to return"),
        skip: int = Query(0, description="Number of tasks to skip")
):
    """List all tasks with optional filtering and pagination."""
    # Build query filter
    query = {}
    if status:
        query["status"] = status
    if platform:
        query["platform"] = platform

    # Get total count
    total_count = tasks_collection.count_documents(query)

    # Get paginated tasks
    tasks_cursor = tasks_collection.find(query).sort("created_at", -1).skip(skip).limit(limit)

    # Convert to list and prepare response
    tasks_list = []
    for task in tasks_cursor:
        # Remove MongoDB ObjectId
        if "_id" in task:
            del task["_id"]

        # Format timestamps for better readability
        if "created_at" in task:
            task["created_at"] = task["created_at"].timestamp()
        if "updated_at" in task:
            task["updated_at"] = task["updated_at"].timestamp()

        tasks_list.append(task)

    return {
        "tasks": tasks_list,
        "total": total_count,
        "limit": limit,
        "skip": skip,
        "has_more": (skip + limit) < total_count
    }


@app.delete("/task/{task_id}")
async def delete_task(task_id: str):
    """Delete a task and its associated files."""
    task_info = tasks_collection.find_one({"task_id": task_id})

    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")

    deleted_files = []

    # Try to delete video file if exists
    if "result" in task_info and "video_path" in task_info["result"]:
        video_path = task_info["result"]["video_path"]
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
                deleted_files.append(os.path.basename(video_path))
            except Exception as e:
                logger.error(f"Error deleting video file: {e}")

    # Try to delete metadata file if exists
    if "result" in task_info and "metadata" in task_info["result"] and "metadata_path" in task_info["result"][
        "metadata"]:
        metadata_path = task_info["result"]["metadata"]["metadata_path"]
        if os.path.exists(metadata_path):
            try:
                os.remove(metadata_path)
                deleted_files.append(os.path.basename(metadata_path))
            except Exception as e:
                logger.error(f"Error deleting metadata file: {e}")

    # Check if there's a running Celery task to revoke
    if "celery_task_id" in task_info and task_info.get("status") in ["queued", "running"]:
        try:
            celery_app.control.revoke(task_info["celery_task_id"], terminate=True)
            logger.info(f"Revoked Celery task: {task_info['celery_task_id']}")
        except Exception as e:
            logger.error(f"Error revoking Celery task: {e}")

    # Remove from MongoDB
    tasks_collection.delete_one({"task_id": task_id})

    return {
        "message": f"Task {task_id} deleted successfully",
        "deleted_files": deleted_files
    }


@app.get("/platforms")
async def list_platforms():
    """List all supported platforms and their configurations."""
    from .config import PLATFORM_CONFIGS
    return {"platforms": list(PLATFORM_CONFIGS.keys()), "configs": PLATFORM_CONFIGS}


@app.get("/config")
async def get_config():
    """Get current configuration."""
    # Return a sanitized version without API keys
    config_dict = config.model_dump()

    # Remove sensitive information
    if "llm" in config_dict and "api_key" in config_dict["llm"]:
        config_dict["llm"]["api_key"] = "***" if config_dict["llm"]["api_key"] else ""

    if "tts" in config_dict and "api_key" in config_dict["tts"]:
        config_dict["tts"]["api_key"] = "***" if config_dict["tts"]["api_key"] else ""

    if "image_gen" in config_dict and "api_key" in config_dict["image_gen"]:
        config_dict["image_gen"]["api_key"] = "***" if config_dict["image_gen"]["api_key"] else ""

    return {"config": config_dict}


@app.get("/health")
async def health_check():
    """Extended health check endpoint."""
    health_info = {
        "status": "healthy",
        "api_version": "1.0.0",
        "tasks": {
            "total": tasks_collection.count_documents({}),
            "running": tasks_collection.count_documents({"status": "running"}),
            "queued": tasks_collection.count_documents({"status": "queued"}),
            "completed": tasks_collection.count_documents({"status": "completed"}),
            "failed": tasks_collection.count_documents({"status": "failed"})
        },
        "config": {
            "platform": config.platform,
            "llm_model": config.llm.model,
            "image_provider": config.image_gen.provider,
            "tts_provider": config.tts.provider
        }
    }

    # Check if required API keys are present
    api_keys_status = config.validate_api_keys()
    health_info["api_keys_configured"] = api_keys_status

    # Check if output directories exist and are writable
    storage_check = all([
        os.access(videos_dir, os.W_OK),
        os.access(metadata_dir, os.W_OK),
        os.access(temp_dir, os.W_OK)
    ])
    health_info["storage_writable"] = storage_check

    # Check connection to Redis
    try:
        celery_inspect = celery_app.control.inspect()
        ping_response = celery_inspect.ping()
        health_info["redis_status"] = "connected" if ping_response else "no_workers"

        if ping_response:
            active = celery_inspect.active()
            health_info["worker_status"] = {
                "online": len(ping_response),
                "active_tasks": sum(len(tasks) for tasks in active.values()) if active else 0
            }
    except Exception as e:
        health_info["redis_status"] = "error"
        health_info["redis_error"] = str(e)

    # Check MongoDB connection
    try:
        mongo_client.admin.command('ping')
        health_info["mongodb_status"] = "connected"
    except Exception as e:
        health_info["mongodb_status"] = "error"
        health_info["mongodb_error"] = str(e)

    # Set overall status
    if (not api_keys_status or
            not storage_check or
            health_info.get("redis_status") != "connected" or
            health_info.get("mongodb_status") != "connected"):
        health_info["status"] = "degraded"

    return health_info