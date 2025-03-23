import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import time

# Add the project directory to the path so we can import modules properly
project_dir = Path(__file__).parent.parent
sys.path.append(str(project_dir))

# Import the necessary modules
from video_generator.pipeline.video_assembler import VideoAssembler
from video_generator.config import PipelineConfig, OUTPUT_DIR
from loguru import logger


def create_mock_config():
    """Create a mock configuration for testing"""
    # Use the default config
    config = PipelineConfig()

    # Set specific values we need for testing
    config.platform = "tiktok"
    config.platform_config.aspect_ratio = "9:16"
    config.visual.color_scheme = "vibrant"
    config.visual.text_animation = True
    config.visual.motion_effects = True

    return config


def create_test_state():
    """Create a test state with sample scenes to process"""
    # Use absolute paths for the media assets
    media_dir = Path(OUTPUT_DIR) / "media_assets"

    # Sample scene data
    scenes = [
        {
            "scene_index": 0,
            "image_path": str(media_dir / "image_0_fb5ddb5be8_c0.png"),
            "voice_path": str(media_dir / "voice_0_444947b46a.mp3"),
            "text": "Did you know your aura can make you 10x more attractive?",
            "text_position": "center",
            "effect": "glow",
            "transition": "fade",
            "duration": 5,
            "success": True
        },
        {
            "scene_index": 1,
            "image_path": str(media_dir / "image_1_27d28c349f_c0.png"),
            "voice_path": str(media_dir / "voice_1_ac4f409386.mp3"),
            "text": "Women with a radiant aura are often seen as the most beautiful.",
            "text_position": "bottom",
            "effect": "text_glow",
            "transition": "fade",
            "duration": 5,
            "success": True
        },
        {
            "scene_index": 2,
            "image_path": str(media_dir / "image_2_c9ee883831_c0.png"),
            "voice_path": str(media_dir / "voice_2_9dc777d3e6.mp3"),
            "text": "This isn't just about looks; it's about the energy they emit.",
            "text_position": "top",
            "effect": "aura_expansion",
            "transition": "fade",
            "duration": 5,
            "success": True
        },
        {
            "scene_index": 3,
            "image_path": str(media_dir / "image_3_fbc908bfc7_c0.png"),
            "voice_path": str(media_dir / "voice_3_684a4bafde.mp3"),
            "text": "A strong, positive aura can enhance your natural beauty and attract positivity.",
            "text_position": "center",
            "effect": "fade",
            "transition": "fade",
            "duration": 5,
            "success": True
        }
    ]

    # Verify that files exist
    for scene in scenes:
        if not os.path.exists(scene["image_path"]):
            logger.error(f"Image file not found: {scene['image_path']}")
            sys.exit(1)
        if not os.path.exists(scene["voice_path"]):
            logger.error(f"Audio file not found: {scene['voice_path']}")
            sys.exit(1)

    # Create a state dictionary with the test scenes
    state = {
        "processed_scenes": scenes,
        "script": {
            "hook": "Did you know your aura can make you 10x more attractive?"
        }
    }

    return state


def run():
    """Run a test of the video assembler"""
    print("Starting video assembler test...")

    # Create a mock config and test state
    config = create_mock_config()
    state = create_test_state()

    # Initialize the video assembler
    assembler = VideoAssembler(config)

    # Process the test state
    start_time = time.time()
    result = assembler.assemble_video(state)
    end_time = time.time()

    # Check the result
    if result.get("success", False):
        print(f"✅ Video assembled successfully in {end_time - start_time:.2f} seconds!")
        print(f"Output path: {result['video_path']}")
        print(f"Video duration: {result['duration']:.2f} seconds")
        print(f"Resolution: {result['resolution']}")
    else:
        print(f"❌ Video assembly failed: {result.get('error', 'Unknown error')}")

    return result


if __name__ == "__main__":
    result = run()