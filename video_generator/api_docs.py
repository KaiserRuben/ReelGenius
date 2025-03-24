"""API Documentation module for ReelGenius."""
from typing import Dict, Any

# Standard response format documentation
STANDARD_RESPONSE_FORMAT = {
    "description": "All API endpoints return a standardized response format",
    "structure": {
        "status": "Status of the response: 'success', 'error', or 'warning'",
        "data": "Main response data - structure varies by endpoint",
        "message": "Human-readable message explaining the response (optional)",
        "error": "Error message in case of failure (optional)",
        "meta": "Additional metadata about the response (optional)"
    },
    "examples": {
        "success": {
            "status": "success",
            "data": {"user_id": "123", "username": "example_user"},
            "message": "User created successfully"
        },
        "error": {
            "status": "error",
            "error": "Invalid input",
            "message": "Username already exists",
            "data": {"field": "username"}
        }
    }
}

# Video generation workflow
VIDEO_GENERATION_WORKFLOW = {
    "description": "The video generation process flow",
    "steps": [
        {
            "step": 1,
            "name": "Request Submission",
            "endpoint": "POST /generate",
            "description": "Submit content for video generation",
            "data_provided": ["content", "platform", "voice_gender"]
        },
        {
            "step": 2,
            "name": "Status Polling",
            "endpoint": "GET /status/{task_id}",
            "description": "Check progress of video generation",
            "data_returned": ["status", "progress", "estimated_time"]
        },
        {
            "step": 3,
            "name": "Completion",
            "endpoint": "GET /status/{task_id}",
            "description": "Get full details once task is completed",
            "data_returned": ["video URL", "metadata", "scene information"]
        },
        {
            "step": 4,
            "name": "Media Access",
            "endpoints": ["GET /video/{task_id}", "GET /media/image/{task_id}/{scene_index}"],
            "description": "Access generated media files"
        }
    ]
}

# API Endpoints documentation
API_ENDPOINTS = {
    "health": {
        "path": "/",
        "method": "GET",
        "description": "Health check endpoint",
        "parameters": None,
        "returns": "API status information"
    },
    "generate_video": {
        "path": "/generate",
        "method": "POST",
        "description": "Generate a video from content",
        "parameters": {
            "content": "Text content to generate video from",
            "platform": "Target platform (tiktok, youtube_shorts, instagram_reels)",
            "voice_gender": "Preferred voice gender (male, female)",
            "config_overrides": "Optional configuration overrides"
        },
        "returns": "Task ID and queued status"
    },
    "task_status": {
        "path": "/status/{task_id}",
        "method": "GET",
        "description": "Get status of a video generation task",
        "parameters": {
            "task_id": "Task ID from generate request",
            "include_scene_details": "Include detailed scene information (boolean)"
        },
        "returns": "Comprehensive task status and progress"
    },
    "get_video": {
        "path": "/video/{task_id}",
        "method": "GET",
        "description": "Get the generated video file",
        "parameters": {
            "task_id": "Task ID from generate request"
        },
        "returns": "Video file (MP4)"
    },
    "get_metadata": {
        "path": "/metadata/{task_id}",
        "method": "GET",
        "description": "Get video metadata",
        "parameters": {
            "task_id": "Task ID from generate request",
            "include_scene_data": "Include detailed scene data (boolean)"
        },
        "returns": "Video metadata and scene information"
    },
    "get_scenes": {
        "path": "/scenes/{task_id}",
        "method": "GET",
        "description": "Get all scenes with media info",
        "parameters": {
            "task_id": "Task ID from generate request"
        },
        "returns": "Scene information with media URLs"
    },
    "get_scene_image": {
        "path": "/media/image/{task_id}/{scene_index}",
        "method": "GET",
        "description": "Get image for a specific scene",
        "parameters": {
            "task_id": "Task ID from generate request",
            "scene_index": "Scene index or 'hook' for hook image"
        },
        "returns": "Image file (JPEG/PNG)"
    },
    "get_scene_audio": {
        "path": "/media/audio/{task_id}/{scene_index}",
        "method": "GET",
        "description": "Get audio for a specific scene",
        "parameters": {
            "task_id": "Task ID from generate request",
            "scene_index": "Scene index or 'hook' for hook audio"
        },
        "returns": "Audio file (MP3)"
    }
}

# Frontend integration guidelines
FRONTEND_INTEGRATION = {
    "polling_recommendations": {
        "initial_delay": "500ms",
        "interval": "2000ms (2 seconds)",
        "backoff": "Increase interval to 5000ms after 30 seconds",
        "max_interval": "10000ms (10 seconds)"
    },
    "request_handling": {
        "error_retry": "Implement exponential backoff for failed requests",
        "timeout": "Set request timeout to 30 seconds for media files, 10 seconds for API calls"
    },
    "ui_patterns": {
        "progress_indicator": "Show progress percentage and estimated time remaining",
        "file_preview": "Display thumbnails for scenes with hover preview",
        "video_player": "Use native HTML5 video player for best compatibility"
    },
    "data_caching": {
        "task_status": "Cache task status responses to reduce polling load",
        "media_urls": "Store media URLs in local state once retrieved",
        "invalidation": "Invalidate cache when new data is received"
    }
}

# Data structures
DATA_STRUCTURES = {
    "TaskStatusResponse": {
        "task_id": "Unique identifier for the task",
        "status": "Current status (queued, running, completed, failed)",
        "progress": "Progress percentage (0.0 to 1.0)",
        "result": "Result data for completed tasks",
        "error": "Error message if task failed",
        "platform": "Target platform for the video",
        "content_summary": "Summary of the content",
        "created_at": "Timestamp when task was created",
        "updated_at": "Timestamp of last status update",
        "execution_time": "Total processing time in seconds"
    },
    "SceneMediaInfo": {
        "image_path": "Path to the scene image file",
        "voice_path": "Path to the scene audio file",
        "text": "Text content for the scene",
        "duration": "Duration of the scene in seconds",
        "transition": "Transition type to the next scene",
        "image_prompt": "Prompt used to generate the image",
        "visual_description": "Description of the visual content"
    },
    "VideoMetadata": {
        "title": "Generated video title",
        "description": "Video description",
        "hashtags": "List of relevant hashtags",
        "category": "Content category",
        "platform": "Target platform",
        "duration": "Video duration in seconds",
        "resolution": "Video resolution (e.g. 1080x1920)",
        "framerate": "Video frame rate in FPS",
        "file_size": "File size in bytes"
    }
}

# Complete API Documentation
API_DOCUMENTATION = {
    "title": "ReelGenius API Documentation",
    "version": "1.0.0",
    "standard_response_format": STANDARD_RESPONSE_FORMAT,
    "workflow": VIDEO_GENERATION_WORKFLOW,
    "endpoints": API_ENDPOINTS,
    "data_structures": DATA_STRUCTURES,
    "frontend_integration": FRONTEND_INTEGRATION
}

def get_documentation() -> Dict[str, Any]:
    """Get full API documentation."""
    return API_DOCUMENTATION