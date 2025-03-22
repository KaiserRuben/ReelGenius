
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Union, Literal
import os
import json
import shutil
from pathlib import Path
import time
import uuid
from loguru import logger

from .config import PipelineConfig, PlatformType
from .pipeline.pipeline import VideoPipeline

app = FastAPI(title="AI Video Generator API")

# Initialize configuration
config = PipelineConfig()

# Ensure output directories exist
output_dir = Path(__file__).parent.parent / "output"
videos_dir = output_dir / "videos"
metadata_dir = output_dir / "metadata"
os.makedirs(videos_dir, exist_ok=True)
os.makedirs(metadata_dir, exist_ok=True)

# Storage for background tasks
tasks = {}

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

# Background task runner
def generate_video_task(task_id: str, content: str, platform: PlatformType = "tiktok", config_overrides: Optional[Dict[str, Any]] = None):
    """Run video generation as a background task."""
    try:
        tasks[task_id]["status"] = "running"
        tasks[task_id]["progress"] = 0.1
        
        # Update config
        task_config = PipelineConfig(platform=platform)
        
        if config_overrides:
            # Update with overrides
            config_dict = task_config.model_dump()
            
            # Apply nested updates
            for key, value in config_overrides.items():
                if key in config_dict:
                    if isinstance(value, dict) and isinstance(config_dict[key], dict):
                        # Nested update
                        config_dict[key].update(value)
                    else:
                        # Direct update
                        config_dict[key] = value
            
            # Create updated config
            task_config = PipelineConfig(**config_dict)
        
        # Initialize pipeline with task-specific config
        task_pipeline = VideoPipeline(task_config)
        
        # Run pipeline
        tasks[task_id]["progress"] = 0.2
        result = task_pipeline.run(content)
        
        tasks[task_id]["progress"] = 1.0
        tasks[task_id]["status"] = "completed" if result.get("success", False) else "failed"
        tasks[task_id]["result"] = result
        
        if not result.get("success", False):
            tasks[task_id]["error"] = result.get("error", "Unknown error")
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["progress"] = 1.0

# API Routes
@app.get("/")
async def root():
    return {"message": "AI Video Generator API", "status": "running"}

@app.post("/generate", response_model=VideoGenerationResponse)
async def generate_video(
    request: VideoGenerationRequest, 
    background_tasks: BackgroundTasks
):
    """Generate a video from content."""
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Initialize task
    tasks[task_id] = {
        "status": "queued",
        "progress": 0,
        "created_at": time.time()
    }
    
    # Start background task
    background_tasks.add_task(
        generate_video_task, 
        task_id, 
        request.content,
        request.platform,
        request.config_overrides
    )
    
    return VideoGenerationResponse(
        task_id=task_id,
        status="queued",
        message="Video generation task has been queued"
    )

@app.post("/generate-from-file", response_model=VideoGenerationResponse)
async def generate_video_from_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    platform: PlatformType = Form("tiktok"),
    config_overrides: Optional[str] = Form(None)
):
    """Generate a video from an uploaded file."""
    # Save uploaded file temporarily
    temp_file = Path(f"/tmp/{uuid.uuid4()}_{file.filename}")
    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Read content from file
        with open(temp_file, "r") as f:
            content = f.read()
        
        # Parse config overrides if provided
        overrides = json.loads(config_overrides) if config_overrides else None
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Initialize task
        tasks[task_id] = {
            "status": "queued",
            "progress": 0,
            "created_at": time.time()
        }
        
        # Start background task
        background_tasks.add_task(
            generate_video_task, 
            task_id, 
            content,
            platform,
            overrides
        )
        
        return VideoGenerationResponse(
            task_id=task_id,
            status="queued",
            message="Video generation task has been queued"
        )
    finally:
        # Clean up temporary file
        if temp_file.exists():
            temp_file.unlink()

@app.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get the status of a video generation task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = tasks[task_id]
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task_info["status"],
        progress=task_info.get("progress"),
        result=task_info.get("result"),
        error=task_info.get("error")
    )

@app.get("/video/{task_id}")
async def get_video(task_id: str):
    """Get the generated video for a completed task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = tasks[task_id]
    
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
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = tasks[task_id]
    
    if task_info["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Task is not completed: {task_info['status']}")
    
    if "result" not in task_info or "metadata" not in task_info["result"]:
        raise HTTPException(status_code=400, detail="Metadata not found in task result")
    
    return task_info["result"]["metadata"]

@app.get("/tasks")
async def list_tasks(
    status: Optional[str] = Query(None, description="Filter by status"),
    platform: Optional[PlatformType] = Query(None, description="Filter by platform"),
    limit: int = Query(10, description="Maximum number of tasks to return")
):
    """List all tasks with optional filtering."""
    filtered_tasks = []
    
    for task_id, task_info in tasks.items():
        if status is not None and task_info["status"] != status:
            continue
            
        if platform is not None and task_info.get("platform") != platform:
            continue
            
        filtered_tasks.append({
            "task_id": task_id,
            "status": task_info["status"],
            "progress": task_info.get("progress"),
            "created_at": task_info["created_at"],
            "platform": task_info.get("platform", "unknown"),
            "error": task_info.get("error")
        })
    
    # Sort by creation time (newest first) and limit
    filtered_tasks.sort(key=lambda x: x["created_at"], reverse=True)
    filtered_tasks = filtered_tasks[:limit]
    
    return {"tasks": filtered_tasks, "total": len(filtered_tasks)}

@app.delete("/task/{task_id}")
async def delete_task(task_id: str):
    """Delete a task and its associated files."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = tasks[task_id]
    
    # Try to delete video file if exists
    if "result" in task_info and "video_path" in task_info["result"]:
        video_path = task_info["result"]["video_path"]
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception as e:
                logger.error(f"Error deleting video file: {e}")
    
    # Try to delete metadata file if exists
    if "result" in task_info and "metadata" in task_info["result"] and "metadata_path" in task_info["result"]["metadata"]:
        metadata_path = task_info["result"]["metadata"]["metadata_path"]
        if os.path.exists(metadata_path):
            try:
                os.remove(metadata_path)
            except Exception as e:
                logger.error(f"Error deleting metadata file: {e}")
    
    # Remove task from storage
    del tasks[task_id]
    
    return {"message": f"Task {task_id} deleted successfully"}

@app.get("/platforms")
async def list_platforms():
    """List all supported platforms and their configurations."""
    from .config import PLATFORM_CONFIGS
    return {"platforms": list(PLATFORM_CONFIGS.keys()), "configs": PLATFORM_CONFIGS}

@app.get("/config")
async def get_config():
    """Get current configuration."""
    return {"config": config.model_dump()}


