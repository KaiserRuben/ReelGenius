from typing import Dict, List, Any, Optional, Union
import os
import requests
import tempfile
import json
import time
from pathlib import Path
from loguru import logger
import hashlib
import asyncio
import aiohttp
import traceback

from ..config import PipelineConfig, OUTPUT_DIR
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from PIL import Image


class ImageGenerationCandidate:
    """Represents a candidate image for a scene."""

    def __init__(self, path: str, score: float = 0.0):
        self.path = path
        self.score = score

    def __lt__(self, other):
        return self.score < other.score


class MediaGenerator:
    """Generate media assets for videos."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = os.path.join(OUTPUT_DIR, "media_assets")
        os.makedirs(self.output_dir, exist_ok=True)

    async def generate_image(self, prompt: str, index: int) -> List[str]:
        """Generate multiple image candidates based on the prompt."""
        provider = self.config.image_gen.provider
        candidates_count = self.config.image_gen.candidates_per_prompt

        # Create a unique base filename
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:10]
        base_filename = f"image_{index}_{prompt_hash}"

        # Check if any candidates already exist
        existing_candidates = []
        for i in range(candidates_count):
            candidate_path = os.path.join(self.output_dir, f"{base_filename}_c{i}.jpeg")
            if os.path.exists(candidate_path):
                existing_candidates.append(candidate_path)

        # If we already have enough candidates, return them
        if len(existing_candidates) >= candidates_count:
            logger.info(f"Using {len(existing_candidates)} existing image candidates for scene {index}")
            return existing_candidates[:candidates_count]

        # Generate new candidates
        candidates = []

        if provider == "stability":
            try:
                candidates = await self._generate_stability_images(prompt, base_filename, candidates_count)
            except Exception as e:
                logger.error(f"Error generating images with Stability: {e}")
                return []
        else:
            logger.error(f"Unsupported image provider: {provider}")
            return []

        return candidates

    async def _generate_stability_images(self, prompt: str, base_filename: str, count: int) -> List[str]:
        """Generate multiple images using Stability AI SD3 API."""
        api_key = self.config.image_gen.api_key

        if not api_key:
            logger.error("Stability API key not found")
            return []

        # Use new Stability AI SD3 endpoint
        url = "https://api.stability.ai/v2beta/stable-image/generate/sd3"

        headers = {
            "authorization": f"Bearer {api_key}",
            "accept": "image/*"
        }

        # Format prompt for appropriate aspect ratio
        aspect_ratio = self.config.platform_config.aspect_ratio
        tiktok_optimized_prompt = f"{prompt}, {aspect_ratio} aspect ratio"

        # Data for the request
        data = {
            "prompt": tiktok_optimized_prompt,
            "output_format": "jpeg",
        }

        candidates = []

        # Generate candidates in parallel
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(count):
                output_path = os.path.join(self.output_dir, f"{base_filename}_c{i}.jpeg")
                task = asyncio.create_task(self._generate_single_stability_image(
                    session, url, headers, data, output_path
                ))
                tasks.append(task)

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error generating candidate {i}: {result}")
                    continue

                if result:
                    candidates.append(result)

        logger.info(f"Generated {len(candidates)} image candidates")
        return candidates

    async def _generate_single_stability_image(self, session, url, headers, data, output_path):
        """Generate a single image using Stability AI API."""
        try:
            async with session.post(url, headers=headers, files={"none": ''}, data=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Stability API error: {response.status} {error_text}")
                    return None

                # Save the image directly from response content
                image_data = await response.read()
                with open(output_path, "wb") as f:
                    f.write(image_data)

                return output_path
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return None

    async def evaluate_images(self, image_paths: List[str], prompt: str) -> List[ImageGenerationCandidate]:
        """Evaluate and score generated images."""
        if not self.config.image_gen.use_image_evaluation or not image_paths:
            # If evaluation is disabled, return all images with default score
            return [ImageGenerationCandidate(path, 1.0) for path in image_paths]

        try:
            # Basic evaluation based on file size and image dimensions
            # In a real implementation, you would use a vision model for quality evaluation
            candidates = []

            for path in image_paths:
                try:
                    # Check if file exists and has content
                    if not os.path.exists(path) or os.path.getsize(path) < 10000:
                        logger.warning(f"Image file missing or too small: {path}")
                        continue

                    # Calculate a basic score based on file size
                    # In a real implementation, use more sophisticated image quality metrics
                    file_size = os.path.getsize(path)
                    score = min(1.0, file_size / 1000000)  # Normalize to 0-1 range

                    candidates.append(ImageGenerationCandidate(path, score))
                except Exception as e:
                    logger.error(f"Error evaluating image {path}: {e}")

            # Sort by score (highest first)
            candidates.sort(reverse=True)
            return candidates

        except Exception as e:
            logger.error(f"Error evaluating images: {e}")
            # Return all images with default score if evaluation fails
            return [ImageGenerationCandidate(path, 1.0) for path in image_paths]

    async def generate_voice(self, text: str, index: int) -> Optional[str]:
        """Generate voice audio from text."""
        provider = self.config.tts.provider

        # Create a unique filename based on the text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()[:10]
        filename = f"voice_{index}_{text_hash}.mp3"
        output_path = os.path.join(self.output_dir, filename)

        # Check if audio already exists
        if os.path.exists(output_path):
            logger.info(f"Voice audio already exists: {output_path}")
            return output_path

        if provider == "elevenlabs":
            try:
                return await self._generate_elevenlabs_voice(text, output_path)
            except Exception as e:
                logger.error(f"Error generating voice with ElevenLabs: {e}")
                return None
        else:
            logger.error(f"Unsupported TTS provider: {provider}")
            return None

    async def _generate_elevenlabs_voice(self, text: str, output_path: str) -> Optional[str]:
        """Generate voice using ElevenLabs API."""
        api_key = self.config.tts.api_key

        if not api_key:
            logger.error("ElevenLabs API key not found")
            return None

        voice_id = self.config.tts.voice_id

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }

        # Optimize text for platform if needed
        if self.config.tts.optimize_for_platform:
            # Add speech marks for emphasis on keywords
            # This is a simple implementation - could be more sophisticated
            words = text.split()
            if len(words) > 5:
                # Add emphasis to roughly every 5th word
                for i in range(4, len(words), 5):
                    if i < len(words):
                        words[i] = f"<emphasis>{words[i]}</emphasis>"

                text = " ".join(words)

        payload = {
            "text": text,
            "model_id": self.config.tts.model,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0 if self.config.tts.voice_style == "natural" else 0.5,
                "use_speaker_boost": True
            }
        }

        # Adjust speaking rate if needed
        if self.config.tts.speaking_rate != 1.0:
            payload["voice_settings"]["speed"] = self.config.tts.speaking_rate

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"ElevenLabs API error: {response.status} {error_text}")
                    return None

                # Save audio file
                with open(output_path, "wb") as f:
                    f.write(await response.read())

                return output_path

    async def process_scene(self, scene: Dict[str, Any], text: str, index: int) -> Dict[str, Any]:
        """Process a single scene, generating all necessary assets."""
        try:
            # Generate images and voice in parallel
            logger.info(f"Processing scene {index}: Generating assets...")

            image_candidates_task = asyncio.create_task(self.generate_image(scene["image_prompt"], index))
            voice_task = asyncio.create_task(self.generate_voice(text, index))

            # Wait for both tasks to complete
            image_paths, voice_path = await asyncio.gather(image_candidates_task, voice_task)

            # Evaluate and select the best image
            if image_paths:
                candidates = await self.evaluate_images(image_paths, scene["image_prompt"])

                # Select the best image (highest score)
                selected_image = candidates[0].path if candidates else None
            else:
                selected_image = None

            return {
                "scene_index": index,
                "image_path": selected_image,
                "voice_path": voice_path,
                "text_overlay": scene.get("text_overlay"),
                "text_position": scene.get("text_position", "center"),
                "effect": scene.get("effect"),
                "transition": scene.get("transition"),
                "duration": scene.get("duration", 5),
                "all_image_candidates": image_paths
            }
        except Exception as e:
            logger.error(f"Error processing scene {index}: {e}")
            logger.error(traceback.format_exc())
            return {
                "scene_index": index,
                "error": str(e),
                "success": False
            }

    async def process(self, visual_plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process all scenes in the visual plan."""
        script_data = visual_plan_data["script_data"]
        visual_plan = visual_plan_data["visual_plan"]
        script = script_data["script"]
        scenes = visual_plan["scenes"]

        # Process all scenes in parallel
        logger.info(f"Processing {len(scenes)} scenes...")

        tasks = []
        for i, scene in enumerate(scenes):
            # Get corresponding script segment
            if i < len(script["segments"]):
                segment_text = script["segments"][i]["text"]
            else:
                segment_text = f"Scene {i + 1}"

            # Create task for processing scene
            task = asyncio.create_task(self.process_scene(scene, segment_text, i))
            tasks.append(task)

        # Wait for all tasks to complete
        processed_scenes = await asyncio.gather(*tasks)

        # Check for any failures
        failures = [scene for scene in processed_scenes if not scene.get("success", True)]
        if failures:
            logger.warning(f"{len(failures)} scenes failed to process")

        return {
            "script_data": script_data,
            "visual_plan": visual_plan,
            "processed_scenes": processed_scenes,
            "success": len(failures) < len(processed_scenes)  # Succeed if at least some scenes processed
        }