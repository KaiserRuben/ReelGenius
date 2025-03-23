from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Query, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
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
from sqlalchemy.orm import Session as SQLAlchemySession
from datetime import datetime, timedelta
import imghdr
import mimetypes

from .config import PipelineConfig, PlatformType
from .tasks import celery_app, generate_video
from .database import get_session, Task

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


# Dependency to get the database session
def get_db():
    db = get_session()
    try:
        yield db
    finally:
        db.close()


# Models
class VideoGenerationRequest(BaseModel):
    content: str
    platform: Optional[PlatformType] = "tiktok"
    voice_gender: Optional[str] = "male"  # Add voice gender selection (male/female)
    config_overrides: Optional[Dict[str, Any]] = None


class VideoGenerationResponse(BaseModel):
    task_id: str
    status: str
    message: str


class SceneMediaInfo(BaseModel):
    """Detailed information about scene media assets"""
    image_path: Optional[str] = None
    voice_path: Optional[str] = None
    text: Optional[str] = None
    duration: Optional[float] = None
    transition: Optional[str] = None
    image_prompt: Optional[str] = None
    visual_description: Optional[str] = None
    
    
class VideoMetadata(BaseModel):
    """Enhanced video metadata"""
    title: Optional[str] = None
    description: Optional[str] = None
    hashtags: Optional[List[str]] = None
    category: Optional[str] = None  
    platform: Optional[str] = None
    duration: Optional[float] = None
    resolution: Optional[str] = None
    framerate: Optional[int] = None
    file_size: Optional[int] = None
    metadata_path: Optional[str] = None


class CacheStats(BaseModel):
    """Statistics about semantic cache performance"""
    hits: int = 0
    misses: int = 0
    money_saved: float = 0.0

class TaskResult(BaseModel):
    """Enhanced task result with detailed media information"""
    video_path: Optional[str] = None
    metadata: Optional[VideoMetadata] = None
    execution_time: Optional[float] = None
    hook_audio_path: Optional[str] = None
    hook_image_path: Optional[str] = None  # Added hook image path
    processed_scenes: Optional[List[SceneMediaInfo]] = None
    script: Optional[Dict[str, Any]] = None
    visual_plan: Optional[Dict[str, Any]] = None
    content_analysis: Optional[Dict[str, Any]] = None
    cache_stats: Optional[CacheStats] = None
    

class TaskStatusResponse(BaseModel):
    """Enhanced task status with comprehensive information"""
    task_id: str
    status: str
    progress: Optional[float] = None
    result: Optional[TaskResult] = None
    error: Optional[str] = None
    platform: Optional[str] = None
    content_summary: Optional[str] = None
    created_at: Optional[float] = None
    updated_at: Optional[float] = None
    execution_time: Optional[float] = None


# Cleanup function
async def cleanup_old_tasks(db: SQLAlchemySession):
    """Clean up old tasks from database"""
    cutoff_time = datetime.utcnow() - timedelta(seconds=TASK_RETENTION_PERIOD)
    
    # Find old tasks
    old_tasks = db.query(Task).filter(Task.created_at < cutoff_time).all()

    for task in old_tasks:
        logger.info(f"Cleaning up old task: {task.task_id}")

        try:
            # Delete related files if they exist
            if task.result and "video_path" in task.result:
                video_path = task.result["video_path"]
                if os.path.exists(video_path):
                    os.remove(video_path)
                    logger.info(f"Deleted video file: {video_path}")

            # Delete metadata if it exists
            if task.result and "metadata" in task.result and "metadata_path" in task.result["metadata"]:
                metadata_path = task.result["metadata"]["metadata_path"]
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                    logger.info(f"Deleted metadata file: {metadata_path}")
        except Exception as e:
            logger.error(f"Error cleaning up files for task {task.task_id}: {e}")

        # Remove from database
        db.delete(task)
    
    db.commit()


# Schedule periodic cleanup
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(periodic_cleanup())
    logger.info("API server started")


async def periodic_cleanup():
    """Run cleanup periodically"""
    cleanup_interval = int(os.environ.get("TASK_CLEANUP_INTERVAL", "3600"))  # Default 1 hour
    while True:
        db = get_session()
        try:
            await cleanup_old_tasks(db)
        finally:
            db.close()
        await asyncio.sleep(cleanup_interval)


# API Routes
@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "AI Video Generator API", "status": "running", "version": "1.0.0"}


@app.post("/generate", response_model=VideoGenerationResponse)
async def create_video(request: VideoGenerationRequest, db: SQLAlchemySession = Depends(get_db)):
    """Generate a video from content."""
    try:
        # Validate input
        if not request.content or len(request.content.strip()) < 10:
            raise HTTPException(status_code=400, detail="Content must be at least 10 characters long")

        # Generate task ID
        task_id = str(uuid.uuid4())

        # Create initial task in database
        new_task = Task(
            task_id=task_id,
            status="queued",
            progress=0,
            created_at=datetime.utcnow(),
            platform=request.platform,
            voice_gender=request.voice_gender,
            content_summary=request.content[:100] + "..." if len(request.content) > 100 else request.content
        )
        db.add(new_task)
        db.commit()

        # Prepare config overrides
        overrides = request.config_overrides or {}
        
        # Set voice_id based on gender preference
        if request.voice_gender and request.voice_gender.lower() == "female":
            overrides.update({"tts": {"voice_id": "Xb7hH8MSUJpSbSDYk0k2"}})
        else:  # default to male
            overrides.update({"tts": {"voice_id": "7fbQ7yJuEo56rYjrYaEh"}})

        # Start Celery task
        celery_result = generate_video.apply_async(
            args=[task_id, request.content, request.platform, overrides],
            task_id=task_id
        )

        # Update database with Celery task ID
        db.query(Task).filter_by(task_id=task_id).update(
            {"celery_task_id": celery_result.id}
        )
        db.commit()

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
        voice_gender: str = Form("male"),
        config_overrides: Optional[str] = Form(None),
        db: SQLAlchemySession = Depends(get_db)
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

        # Store initial task info in database
        new_task = Task(
            task_id=task_id,
            status="queued",
            progress=0,
            created_at=datetime.utcnow(),
            platform=platform,
            voice_gender=voice_gender,
            content_source=file.filename,
            content_summary=content[:100] + "..." if len(content) > 100 else content
        )
        db.add(new_task)
        db.commit()

        # Process voice gender from the parameter
        if not overrides:
            overrides = {}
        
        # Set voice_id based on gender preference
        if voice_gender.lower() == "female":
            overrides.update({"tts": {"voice_id": "Xb7hH8MSUJpSbSDYk0k2"}})
        else:  # default to male
            overrides.update({"tts": {"voice_id": "7fbQ7yJuEo56rYjrYaEh"}})
            
        # Start Celery task
        celery_result = generate_video.apply_async(
            args=[task_id, content, platform, overrides],
            task_id=task_id
        )

        # Update database with Celery task ID
        db.query(Task).filter_by(task_id=task_id).update(
            {"celery_task_id": celery_result.id}
        )
        db.commit()

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
async def get_task_status(task_id: str, db: SQLAlchemySession = Depends(get_db)):
    """Get the status of a video generation task."""
    # First check database
    task = db.query(Task).filter_by(task_id=task_id).first()

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Convert to dictionary for easier handling
    task_info = task.to_dict()

    # If task has a Celery ID and isn't completed yet, check Celery for latest status
    if task.celery_task_id and task.status not in ["completed", "failed"]:
        try:
            celery_task = celery_app.AsyncResult(task.celery_task_id)

            if celery_task.state == 'SUCCESS' and celery_task.result:
                # Update from Celery result
                result = celery_task.result
                
                # Update task in database
                task.status = result.get("status", "completed")
                task.progress = result.get("progress", 1.0)
                task.result = result.get("result", {})
                task.error = result.get("error")
                task.execution_time = result.get("execution_time")
                task.updated_at = datetime.utcnow()
                
                db.commit()
                
                # Refresh task_info
                task_info = task.to_dict()

            elif celery_task.state == 'FAILURE':
                # Update with failure info
                task.status = "failed"
                task.progress = 1.0
                task.error = str(celery_task.result)
                task.updated_at = datetime.utcnow()
                
                db.commit()
                
                # Refresh task_info
                task_info = task.to_dict()

            # Pending and progress states are already updated by the task itself

        except Exception as e:
            logger.error(f"Error checking Celery task status: {e}")
            # Continue with current database status
    
    # Format result data for enhanced response if available
    result_data = None
    if "result" in task_info and task_info["result"]:
        result = task_info["result"]
        
        # Extract scene media info if available
        processed_scenes = []
        if "processed_scenes" in result:
            for scene in result.get("processed_scenes", []):
                # Add file sizes if available
                image_size = None
                voice_size = None
                if scene.get("image_path") and os.path.exists(scene["image_path"]):
                    image_size = os.path.getsize(scene["image_path"])
                if scene.get("voice_path") and os.path.exists(scene["voice_path"]):
                    voice_size = os.path.getsize(scene["voice_path"])
                
                processed_scenes.append(SceneMediaInfo(
                    image_path=scene.get("image_path"),
                    voice_path=scene.get("voice_path"),
                    text=scene.get("text"),
                    duration=scene.get("precise_duration"),
                    transition=scene.get("transition"),
                    image_prompt=scene.get("image_prompt"),
                    visual_description=scene.get("visual_description")
                ))
        
        # Extract video metadata if available
        metadata = None
        if "metadata" in result:
            meta = result["metadata"]
            # Add video file size if available
            file_size = None
            video_path = result.get("video_path")
            if video_path and os.path.exists(video_path):
                file_size = os.path.getsize(video_path)
                
            metadata = VideoMetadata(
                title=meta.get("title"),
                description=meta.get("description"),
                hashtags=meta.get("hashtags"),
                category=meta.get("category"),
                platform=meta.get("platform"),
                duration=meta.get("duration") or result.get("duration"),
                resolution=meta.get("resolution"),
                framerate=meta.get("framerate", 30),
                file_size=file_size,
                metadata_path=meta.get("metadata_path")
            )
        
        # Check for cache statistics
        cache_stats = None
        if "cache_stats" in result:
            cache_stats = CacheStats(
                hits=result["cache_stats"].get("hits", 0),
                misses=result["cache_stats"].get("misses", 0),
                money_saved=result["cache_stats"].get("money_saved", 0.0)
            )
            
        # Create comprehensive result object
        result_data = TaskResult(
            video_path=result.get("video_path"),
            metadata=metadata,
            execution_time=result.get("execution_time") or task_info.get("execution_time"),
            hook_audio_path=result.get("hook_audio_path"),
            hook_image_path=result.get("hook_image_path"),
            processed_scenes=processed_scenes if processed_scenes else None,
            script=result.get("script"),
            visual_plan=result.get("visual_plan"),
            content_analysis=result.get("content_analysis"),
            cache_stats=cache_stats
        )
    
    # Create enhanced response
    return TaskStatusResponse(
        task_id=task_id,
        status=task_info.get("status", "unknown"),
        progress=task_info.get("progress"),
        result=result_data,
        error=task_info.get("error"),
        platform=task_info.get("platform"),
        content_summary=task_info.get("content_summary"),
        created_at=task_info.get("created_at"),
        updated_at=task_info.get("updated_at"),
        execution_time=task_info.get("execution_time")
    )


@app.get("/video/{task_id}")
async def get_video(task_id: str, db: SQLAlchemySession = Depends(get_db)):
    """Get the generated video for a completed task."""
    task = db.query(Task).filter_by(task_id=task_id).first()

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != "completed":
        raise HTTPException(status_code=400, detail=f"Task is not completed: {task.status}")

    if not task.result or "video_path" not in task.result:
        raise HTTPException(status_code=400, detail="Video path not found in task result")

    video_path = task.result["video_path"]

    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=os.path.basename(video_path)
    )


@app.get("/metadata/{task_id}", response_model=Dict[str, Any])
async def get_metadata(
    task_id: str, 
    include_scene_data: bool = Query(False, description="Include detailed scene data"),
    db: SQLAlchemySession = Depends(get_db)
):
    """Get the metadata for a completed task with enhanced details."""
    task = db.query(Task).filter_by(task_id=task_id).first()

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != "completed":
        raise HTTPException(status_code=400, detail=f"Task is not completed: {task.status}")

    if not task.result or "metadata" not in task.result:
        raise HTTPException(status_code=400, detail="Metadata not found in task result")
        
    # Get basic metadata
    result = task.result
    metadata = result["metadata"]
    
    # Add additional information from result
    enhanced_metadata = dict(metadata)
    
    # Add video path and file info
    if "video_path" in result:
        video_path = result["video_path"]
        enhanced_metadata["video_path"] = video_path
        
        # Add video file size if available
        if os.path.exists(video_path):
            enhanced_metadata["file_size_bytes"] = os.path.getsize(video_path)
    
    # Add generation info
    enhanced_metadata["generation_time"] = result.get("execution_time")
    
    # Add cache statistics if available
    if "cache_stats" in result:
        enhanced_metadata["cache_stats"] = result["cache_stats"]
        enhanced_metadata["money_saved"] = result["cache_stats"].get("money_saved", 0.0)
    
    # Add scene information if requested
    if include_scene_data and "processed_scenes" in result:
        scene_data = []
        
        for i, scene in enumerate(result["processed_scenes"]):
            scene_info = {
                "index": i,
                "text": scene.get("text", ""),
                "duration": scene.get("precise_duration")
            }
            
            # Add media paths and generate URLs if available
            if scene.get("image_path") and os.path.exists(scene["image_path"]):
                scene_info["image_path"] = scene["image_path"]
                scene_info["image_filename"] = os.path.basename(scene["image_path"])
                scene_info["image_url"] = f"/api/media/image/{task_id}/{i}"
                scene_info["image_size_bytes"] = os.path.getsize(scene["image_path"])
                
            if scene.get("voice_path") and os.path.exists(scene["voice_path"]):
                scene_info["voice_path"] = scene["voice_path"]
                scene_info["voice_filename"] = os.path.basename(scene["voice_path"])
                scene_info["voice_url"] = f"/api/media/audio/{task_id}/{i}"
                scene_info["voice_size_bytes"] = os.path.getsize(scene["voice_path"])
                
            # Add other scene data
            if "image_prompt" in scene:
                scene_info["image_prompt"] = scene["image_prompt"]
                
            if "visual_description" in scene:
                scene_info["visual_description"] = scene["visual_description"]
                
            scene_data.append(scene_info)
            
        enhanced_metadata["scenes"] = scene_data
        enhanced_metadata["scene_count"] = len(scene_data)
        
    # Add script data
    if "script" in result:
        enhanced_metadata["script"] = result["script"]
    
    # Add hook assets if available
    if "hook_audio_path" in result and os.path.exists(result["hook_audio_path"]):
        enhanced_metadata["hook_audio_path"] = result["hook_audio_path"]
        enhanced_metadata["hook_audio_filename"] = os.path.basename(result["hook_audio_path"])
        enhanced_metadata["hook_audio_url"] = f"/api/media/audio/{task_id}/hook"
        enhanced_metadata["hook_audio_size_bytes"] = os.path.getsize(result["hook_audio_path"])
        
    if "hook_image_path" in result and os.path.exists(result["hook_image_path"]):
        enhanced_metadata["hook_image_path"] = result["hook_image_path"]
        enhanced_metadata["hook_image_filename"] = os.path.basename(result["hook_image_path"])
        enhanced_metadata["hook_image_url"] = f"/api/media/image/{task_id}/hook"
        enhanced_metadata["hook_image_size_bytes"] = os.path.getsize(result["hook_image_path"])

    return enhanced_metadata


@app.get("/tasks")
async def list_tasks(
        status: Optional[str] = Query(None, description="Filter by status"),
        platform: Optional[PlatformType] = Query(None, description="Filter by platform"),
        limit: int = Query(10, description="Maximum number of tasks to return"),
        skip: int = Query(0, description="Number of tasks to skip"),
        include_details: bool = Query(False, description="Include additional task details"),
        include_media_info: bool = Query(False, description="Include media file information"),
        db: SQLAlchemySession = Depends(get_db)
):
    """List all tasks with optional filtering, pagination, and enhanced information."""
    # Build query filter
    query = db.query(Task)
    if status:
        query = query.filter(Task.status == status)
    if platform:
        query = query.filter(Task.platform == platform)

    # Get total count
    total_count = query.count()

    # Get paginated tasks
    tasks_query = query.order_by(Task.created_at.desc()).offset(skip).limit(limit)
    tasks = tasks_query.all()

    # Convert to list and prepare response
    tasks_list = []
    for task in tasks:
        # Convert to dictionary
        task_dict = task.to_dict()
            
        # Include enhanced task information if requested
        if include_details and task.status == "completed" and task.result:
            result = task.result
            
            # Add a summary of the video
            task_dict["summary"] = {
                "duration": result.get("duration"),
                "resolution": result.get("resolution"),
                "scene_count": len(result.get("processed_scenes", [])),
                "title": result.get("metadata", {}).get("title"),
                "platform": task.platform
            }
            
            # Add execution metrics
            task_dict["metrics"] = {
                "execution_time": result.get("execution_time") or task.execution_time,
                "start_time": task_dict.get("created_at"),
                "end_time": task_dict.get("updated_at")
            }
            
            # Add URLs for easy access to media
            task_dict["urls"] = {
                "video": f"/video/{task.task_id}",
                "metadata": f"/metadata/{task.task_id}",
                "scenes": f"/scenes/{task.task_id}"
            }
            
            # Add media info if requested
            if include_media_info and "processed_scenes" in result:
                media_info = {
                    "image_count": 0,
                    "audio_count": 0,
                    "total_image_size_bytes": 0,
                    "total_audio_size_bytes": 0,
                    "video_size_bytes": 0
                }
                
                # Count and tally media files
                for scene in result["processed_scenes"]:
                    if scene.get("image_path") and os.path.exists(scene["image_path"]):
                        media_info["image_count"] += 1
                        media_info["total_image_size_bytes"] += os.path.getsize(scene["image_path"])
                        
                    if scene.get("voice_path") and os.path.exists(scene["voice_path"]):
                        media_info["audio_count"] += 1
                        media_info["total_audio_size_bytes"] += os.path.getsize(scene["voice_path"])
                
                # Add video file size
                if "video_path" in result and os.path.exists(result["video_path"]):
                    media_info["video_size_bytes"] = os.path.getsize(result["video_path"])
                    
                # Add hook assets if available
                media_info["has_hook"] = False
                
                if "hook_audio_path" in result and os.path.exists(result["hook_audio_path"]):
                    media_info["has_hook_audio"] = True
                    media_info["has_hook"] = True
                    media_info["audio_count"] += 1
                    media_info["total_audio_size_bytes"] += os.path.getsize(result["hook_audio_path"])
                else:
                    media_info["has_hook_audio"] = False
                    
                if "hook_image_path" in result and os.path.exists(result["hook_image_path"]):
                    media_info["has_hook_image"] = True
                    media_info["has_hook"] = True
                    media_info["image_count"] += 1
                    media_info["total_image_size_bytes"] += os.path.getsize(result["hook_image_path"])
                else:
                    media_info["has_hook_image"] = False
                    
                task_dict["media_info"] = media_info
            
            # Remove the full result to keep the response lighter
            if "result" in task_dict:
                del task_dict["result"]
        
        tasks_list.append(task_dict)

    return {
        "tasks": tasks_list,
        "total": total_count,
        "limit": limit,
        "skip": skip,
        "has_more": (skip + limit) < total_count
    }


@app.delete("/task/{task_id}")
async def delete_task(task_id: str, db: SQLAlchemySession = Depends(get_db)):
    """Delete a task and its associated files."""
    task = db.query(Task).filter_by(task_id=task_id).first()

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    deleted_files = []

    # Try to delete video file if exists
    if task.result and "video_path" in task.result:
        video_path = task.result["video_path"]
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
                deleted_files.append(os.path.basename(video_path))
            except Exception as e:
                logger.error(f"Error deleting video file: {e}")

    # Try to delete metadata file if exists
    if task.result and "metadata" in task.result and "metadata_path" in task.result["metadata"]:
        metadata_path = task.result["metadata"]["metadata_path"]
        if os.path.exists(metadata_path):
            try:
                os.remove(metadata_path)
                deleted_files.append(os.path.basename(metadata_path))
            except Exception as e:
                logger.error(f"Error deleting metadata file: {e}")

    # Check if there's a running Celery task to revoke
    if task.celery_task_id and task.status in ["queued", "running"]:
        try:
            celery_app.control.revoke(task.celery_task_id, terminate=True)
            logger.info(f"Revoked Celery task: {task.celery_task_id}")
        except Exception as e:
            logger.error(f"Error revoking Celery task: {e}")

    # Remove from database
    db.delete(task)
    db.commit()

    return {
        "message": f"Task {task_id} deleted successfully",
        "deleted_files": deleted_files
    }


@app.get("/platforms")
async def list_platforms():
    """List all supported platforms and their configurations."""
    from .config import PLATFORM_CONFIGS
    return {
        "platforms": list(PLATFORM_CONFIGS.keys()), 
        "configs": PLATFORM_CONFIGS,
        "voice_options": {
            "male": "7fbQ7yJuEo56rYjrYaEh",
            "female": "Xb7hH8MSUJpSbSDYk0k2"
        }
    }


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


@app.get("/media/image/{task_id}/{scene_index}")
async def get_scene_image(task_id: str, scene_index: str, db: SQLAlchemySession = Depends(get_db)):
    """Get the image for a specific scene or hook in a completed task."""
    task = db.query(Task).filter_by(task_id=task_id).first()

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != "completed":
        raise HTTPException(status_code=400, detail=f"Task is not completed: {task.status}")

    result = task.result
    if not result:
        raise HTTPException(status_code=400, detail="Result data not found in task")
    
    # Check if requesting hook image
    if scene_index.lower() == "hook":
        if "hook_image_path" not in result or not os.path.exists(result["hook_image_path"]):
            raise HTTPException(status_code=404, detail="Hook image not found")
            
        return FileResponse(
            result["hook_image_path"],
            media_type="image/jpeg",
            filename=os.path.basename(result["hook_image_path"])
        )
    
    # Else handle regular scene images
    if "processed_scenes" not in result:
        raise HTTPException(status_code=400, detail="Scene data not found in task result")
        
    scenes = result["processed_scenes"]
    
    # Handle invalid scene index
    try:
        scene_idx = int(scene_index)
        if scene_idx < 0 or scene_idx >= len(scenes):
            raise HTTPException(status_code=404, detail=f"Scene index {scene_idx} out of range")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid scene index")
        
    scene = scenes[scene_idx]
    
    if "image_path" not in scene or not os.path.exists(scene["image_path"]):
        raise HTTPException(status_code=404, detail="Image not found for this scene")
        
    return FileResponse(
        scene["image_path"],
        media_type="image/jpeg",
        filename=os.path.basename(scene["image_path"])
    )


@app.get("/media/audio/{task_id}/{scene_index}")
async def get_scene_audio(task_id: str, scene_index: str, db: SQLAlchemySession = Depends(get_db)):
    """Get the audio for a specific scene or hook in a completed task."""
    task = db.query(Task).filter_by(task_id=task_id).first()

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != "completed":
        raise HTTPException(status_code=400, detail=f"Task is not completed: {task.status}")

    if not task.result:
        raise HTTPException(status_code=400, detail="Result data not found in task")
        
    result = task.result
    
    # Check if requesting hook audio
    if scene_index.lower() == "hook":
        if "hook_audio_path" not in result or not os.path.exists(result["hook_audio_path"]):
            raise HTTPException(status_code=404, detail="Hook audio not found")
            
        return FileResponse(
            result["hook_audio_path"],
            media_type="audio/mpeg",
            filename=os.path.basename(result["hook_audio_path"])
        )
    
    # Otherwise, process as regular scene audio
    if "processed_scenes" not in result:
        raise HTTPException(status_code=400, detail="Scene data not found in task result")
        
    scenes = result["processed_scenes"]
    
    # Handle invalid scene index
    try:
        scene_idx = int(scene_index)
        if scene_idx < 0 or scene_idx >= len(scenes):
            raise HTTPException(status_code=404, detail=f"Scene index {scene_idx} out of range")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid scene index")
        
    scene = scenes[scene_idx]
    
    if "voice_path" not in scene or not os.path.exists(scene["voice_path"]):
        raise HTTPException(status_code=404, detail="Audio not found for this scene")
        
    return FileResponse(
        scene["voice_path"],
        media_type="audio/mpeg",
        filename=os.path.basename(scene["voice_path"])
    )


@app.get("/scenes/{task_id}")
async def get_scenes(task_id: str, db: SQLAlchemySession = Depends(get_db)):
    """Get all scenes with media info for a completed task."""
    task = db.query(Task).filter_by(task_id=task_id).first()

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != "completed":
        raise HTTPException(status_code=400, detail=f"Task is not completed: {task.status}")

    if not task.result or "processed_scenes" not in task.result:
        raise HTTPException(status_code=400, detail="Scene data not found in task result")
        
    scenes = task.result["processed_scenes"]
    result = []
    
    # Build enhanced scene info
    for i, scene in enumerate(scenes):
        scene_info = {
            "index": i,
            "text": scene.get("text", ""),
            "duration": scene.get("precise_duration"),
            "transition": scene.get("transition", "crossfade")
        }
        
        # Add media URLs
        if scene.get("image_path") and os.path.exists(scene["image_path"]):
            scene_info["image_url"] = f"/media/image/{task_id}/{i}"
            scene_info["image_size_bytes"] = os.path.getsize(scene["image_path"])
            
        if scene.get("voice_path") and os.path.exists(scene["voice_path"]):
            scene_info["audio_url"] = f"/media/audio/{task_id}/{i}"
            scene_info["audio_size_bytes"] = os.path.getsize(scene["voice_path"])
            
        # Add image generation info
        if "image_prompt" in scene:
            scene_info["image_prompt"] = scene["image_prompt"]
            
        if "visual_description" in scene:
            scene_info["visual_description"] = scene["visual_description"]
            
        result.append(scene_info)
        
    return {
        "task_id": task_id,
        "scene_count": len(result),
        "scenes": result,
        # Add script if available
        "script": task.result.get("script")
    }


@app.get("/cache/stats")
async def cache_stats():
    """Get statistics about the semantic cache."""
    from video_generator.semantic_cache.cache import _cache_manager
    
    try:
        # Get Redis info if available
        redis_info = None
        try:
            redis_cache = _cache_manager.redis_cache
            if hasattr(redis_cache, 'info'):
                redis_info = redis_cache.info()
        except Exception as e:
            redis_info = {"error": str(e)}
        
        # Create summarized stats
        stats = {
            "status": "active",
            "engine": "redis" if redis_info else "in-memory",
            "embedding_model": _cache_manager.model.get_sentence_embedding_dimension(),
            "embedding_cache_size": len(_cache_manager._embedding_cache),
            "max_embedding_cache_size": _cache_manager._max_cache_size,
            "redis_info": redis_info
        }
        
        # Add usage info
        tasks_data = []
        if os.path.exists("cache_keys.csv"):
            with open("cache_keys.csv", "r") as f:
                lines = f.readlines()
                if lines:
                    stats["total_cache_entries"] = len(lines)
                    # Get last 10 entries
                    recent = lines[-10:] if len(lines) >= 10 else lines
                    for line in recent:
                        parts = line.strip().split(",", 4)
                        if len(parts) >= 5:
                            tasks_data.append({
                                "timestamp": parts[0],
                                "key_prefix": parts[1],
                                "hash": parts[2],
                                "similarity_enabled": parts[3].lower() == "true",
                                "content": parts[4][:100] + "..." if len(parts[4]) > 100 else parts[4]
                            })
                    stats["recent_entries"] = tasks_data
        
        return stats
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/health")
async def health_check(db: SQLAlchemySession = Depends(get_db)):
    """Extended health check endpoint."""
    health_info = {
        "status": "healthy",
        "api_version": "1.0.0",
        "tasks": {
            "total": db.query(Task).count(),
            "running": db.query(Task).filter_by(status="running").count(),
            "queued": db.query(Task).filter_by(status="queued").count(),
            "completed": db.query(Task).filter_by(status="completed").count(),
            "failed": db.query(Task).filter_by(status="failed").count()
        },
        "config": {
            "platform": config.platform,
            "llm_model": config.llm.model,
            "image_provider": config.image_gen.provider,
            "tts_provider": config.tts.provider
        },
        "semantic_cache": {
            "status": "enabled"
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

    # Check PostgreSQL connection
    try:
        # Simply executing a query confirms connection
        db.execute("SELECT 1")
        health_info["postgres_status"] = "connected"
    except Exception as e:
        health_info["postgres_status"] = "error"
        health_info["postgres_error"] = str(e)

    # Set overall status
    if (not api_keys_status or
            not storage_check or
            health_info.get("redis_status") != "connected" or
            health_info.get("postgres_status") != "connected"):
        health_info["status"] = "degraded"

    return health_info