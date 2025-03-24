from typing import Dict, List, Any, Optional, Union
import os
import requests
import tempfile
import json
import time
from pathlib import Path
from loguru import logger
import hashlib
import traceback
from PIL import Image

from video_generator.config import PipelineConfig, OUTPUT_DIR
# Removed semantic cache imports to avoid segmentation faults


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

        # Set reasonable timeouts for API requests
        self.api_timeout = int(os.environ.get("API_TIMEOUT", "30"))  # Default 30 seconds
        
        # Cache stats tracking
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "money_saved": 0.0  # Estimated cost savings based on image generation API cost
        }

    # Maintaining async interface for compatibility
    async def generate_image(self, prompt: str, index: int) -> List[str]:
        """Generate multiple image candidates based on the prompt."""
        return self._generate_image_sync(prompt, index)

    # Synchronous implementation
    def _generate_image_sync(self, prompt: str, index: int) -> List[str]:
        """Synchronous implementation of image generation."""
        provider = self.config.image_gen.provider

        # Configurable candidates - default to 1 for simpler operation
        # Can be overridden via config if multiple candidates are desired
        candidates_count = getattr(self.config.image_gen, 'candidates_per_prompt', 1)

        # Create a unique base filename
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:10]
        base_filename = f"image_{index}_{prompt_hash}"
        
        # Semantic cache disabled to prevent segmentation faults
        logger.info(f"CACHE DISABLED: Semantic cache skipped for image generation of scene {index}")
        
        # Cache miss tracking for statistics
        self.cache_stats["misses"] += 1

        # Check if any candidates already exist
        existing_candidates = []
        for i in range(candidates_count):
            candidate_path = os.path.join(self.output_dir, f"{base_filename}_c{i}.jpeg")
            if os.path.exists(candidate_path):
                existing_candidates.append(candidate_path)

        # If we already have enough candidates, return them
        if len(existing_candidates) >= candidates_count:
            logger.info(f"Using {len(existing_candidates)} existing image candidates for scene {index}")
            # Semantic cache disabled to prevent segmentation faults
            logger.info(f"CACHE DISABLED: Not storing candidates in semantic cache for scene {index}")
            return existing_candidates[:candidates_count]

        # Generate new candidates
        candidates = []

        if provider == "stability":
            try:
                candidates = self._generate_stability_images(prompt, base_filename, candidates_count)
                # Semantic cache disabled to prevent segmentation faults
                if candidates:
                    logger.info(f"CACHE DISABLED: Not storing candidates in semantic cache for provider {provider}")
            except Exception as e:
                logger.error(f"Error generating images with Stability: {str(e)}")
                logger.error(traceback.format_exc())
                return []
        else:
            logger.error(f"Unsupported image provider: {provider}")
            return []

        return candidates

    def _generate_stability_images(self, prompt: str, base_filename: str, count: int) -> List[str]:
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

        candidates = []

        # Process candidates sequentially
        for i in range(count):
            output_path = os.path.join(self.output_dir, f"{base_filename}_c{i}.png")
            result = self._generate_single_stability_image(url, headers, tiktok_optimized_prompt, output_path)

            if result:
                candidates.append(result)

        logger.info(f"Generated {len(candidates)} image candidates")
        return candidates

    # Removed @cached decorator to prevent segmentation faults
    def _generate_single_stability_image(self, url, headers, prompt, output_path):
        """Generate a single image using Stability AI API."""
        try:
            # Check if output file already exists (might be from a previous run)
            if os.path.exists(output_path):
                try:
                    img = Image.open(output_path)
                    img.verify()  # Verify it's a valid image
                    logger.info(f"Using existing image at {output_path}")
                    return output_path
                except Exception:
                    # Image exists but is invalid, proceed with generation
                    pass
            
            # Create multipart/form-data request
            files = {
                'prompt': (None, prompt),
                'output_format': (None, 'png'),
                'aspect_ratio': (None, self.config.platform_config.aspect_ratio)
            }

            # Use a timeout to prevent hanging requests
            response = requests.post(
                url,
                headers=headers,
                files=files,  # Using 'files' parameter sets content-type as multipart/form-data
                timeout=self.api_timeout
            )

            if response.status_code != 200:
                # Check for rate limiting
                if response.status_code == 429:
                    logger.warning("Rate limit exceeded with Stability API. Waiting before retry...")
                    time.sleep(5)  # Wait 5 seconds before retry

                    # Try again with a simple retry
                    retry_response = requests.post(
                        url,
                        headers=headers,
                        files=files,
                        timeout=self.api_timeout
                    )
                    if retry_response.status_code == 200:
                        with open(output_path, "wb") as f:
                            f.write(retry_response.content)
                        logger.info(f"Generated new image at {output_path} (after retry)")
                        return output_path
                    else:
                        logger.error(f"Stability API retry error: {retry_response.status_code} {retry_response.text}")
                        return None

                # Other error types
                logger.error(f"Stability API error: {response.status_code} {response.text}")
                return None

            # Save the image directly from response content
            with open(output_path, "wb") as f:
                f.write(response.content)

            # Validate the image file
            try:
                img = Image.open(output_path)
                img.verify()  # Verify it's a valid image
                logger.info(f"Generated new image at {output_path}")
                return output_path
            except Exception as e:
                logger.error(f"Generated invalid image: {e}")
                if os.path.exists(output_path):
                    os.remove(output_path)  # Remove invalid file
                return None

        except requests.exceptions.Timeout:
            logger.error(f"Timeout generating image from Stability API")
            return None
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            logger.error(traceback.format_exc())
            return None

    # Maintaining async interface for compatibility
    async def evaluate_images(self, image_paths: List[str], prompt: str) -> List[ImageGenerationCandidate]:
        """Evaluate and score generated images."""
        return self._evaluate_images_sync(image_paths, prompt)

    # Synchronous implementation
    def _evaluate_images_sync(self, image_paths: List[str], prompt: str) -> List[ImageGenerationCandidate]:
        """Synchronous implementation of image evaluation."""
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

                    try:
                        # Check if the image can be opened
                        img = Image.open(path)
                        width, height = img.size

                        # Calculate a score based on multiple factors
                        file_size = os.path.getsize(path)
                        size_score = min(1.0, file_size / 1000000)  # Normalize to 0-1 range

                        # Check image dimensions
                        aspect_ratio = self.config.platform_config.aspect_ratio
                        expected_ratio = [float(x) for x in aspect_ratio.split(":")]
                        expected_ratio = expected_ratio[0] / expected_ratio[1]
                        actual_ratio = width / height

                        # Calculate how close the actual ratio is to expected
                        ratio_diff = abs(expected_ratio - actual_ratio)
                        ratio_score = max(0, 1.0 - ratio_diff * 10)  # Penalize incorrect ratios

                        # Combined score
                        score = (size_score * 0.4) + (ratio_score * 0.6)

                        candidates.append(ImageGenerationCandidate(path, score))
                    except Exception as e:
                        logger.error(f"Error evaluating image {path}: {e}")
                        # If image can't be opened, give it a low score
                        candidates.append(ImageGenerationCandidate(path, 0.1))
                except Exception as e:
                    logger.error(f"Error evaluating image {path}: {e}")

            # Sort by score (highest first)
            candidates.sort(reverse=True)
            return candidates

        except Exception as e:
            logger.error(f"Error evaluating images: {e}")
            logger.error(traceback.format_exc())
            # Return all images with default score if evaluation fails
            return [ImageGenerationCandidate(path, 1.0) for path in image_paths]

    # Generate hook image and voice
    def _generate_hook_assets(self, state: Dict[str, Any]) -> Dict[str, str]:
        """Generate image and voice audio for the hook."""
        result = {
            "hook_audio_path": None,
            "hook_image_path": None
        }
        
        if "script" not in state or "hook" not in state["script"]:
            logger.warning("No hook found in script")
            return result

        hook_text = state["script"]["hook"]

        # Create a unique hash for the hook
        hook_hash = hashlib.md5(hook_text.encode()).hexdigest()[:10]
        
        # Generate hook audio
        audio_filename = f"hook_{hook_hash}.mp3"
        audio_output_path = os.path.join(self.output_dir, audio_filename)

        # Check if hook audio already exists
        if os.path.exists(audio_output_path):
            logger.info(f"Hook audio already exists: {audio_output_path}")
            result["hook_audio_path"] = audio_output_path
        else:
            # Generate hook audio using the same TTS provider as regular voice
            result["hook_audio_path"] = self._generate_voice_sync(hook_text, "hook")
        
        # Generate hook image - create an attention-grabbing image prompt
        image_prompt = f"Attention-grabbing visual for: '{hook_text}'. Bold, vibrant, eye-catching image that represents the hook statement. {self.config.platform_config.aspect_ratio} ratio. Very high quality, engaging and designed to stop scrolling."
        
        # Check for existing hook image
        image_filename = f"hook_{hook_hash}.jpeg"
        image_output_path = os.path.join(self.output_dir, image_filename)
        
        if os.path.exists(image_output_path):
            logger.info(f"Hook image already exists: {image_output_path}")
            result["hook_image_path"] = image_output_path
        else:
            # Generate image using our existing image generation pipeline
            image_paths = self._generate_image_sync(image_prompt, "hook")
            if image_paths:
                # Select the best image
                candidates = self._evaluate_images_sync(image_paths, image_prompt)
                if candidates:
                    result["hook_image_path"] = candidates[0].path
                    logger.info(f"Generated new hook image: {result['hook_image_path']}")
        
        logger.info(f"Hook assets prepared: audio={result['hook_audio_path'] is not None}, image={result['hook_image_path'] is not None}")
        return result

    # Maintaining async interface for compatibility
    async def generate_voice(self, text: str, index: int) -> Optional[str]:
        """Generate voice audio from text."""
        return self._generate_voice_sync(text, index)

    # Synchronous implementation
    def _generate_voice_sync(self, text: str, index: int) -> Optional[str]:
        """Synchronous implementation of voice generation."""
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
                return self._generate_elevenlabs_voice(text, output_path)
            except Exception as e:
                logger.error(f"Error generating voice with ElevenLabs: {e}")
                logger.error(traceback.format_exc())
                return None
        else:
            logger.error(f"Unsupported TTS provider: {provider}")
            return None

    def _generate_elevenlabs_voice(self, text: str, output_path: str) -> Optional[str]:
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

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.api_timeout)

            if response.status_code != 200:
                # Check for rate limiting
                if response.status_code == 429:
                    logger.warning("Rate limit exceeded with ElevenLabs API. Waiting before retry...")
                    time.sleep(5)  # Wait 5 seconds before retry

                    # Try again with a simple retry
                    retry_response = requests.post(url, headers=headers, json=payload, timeout=self.api_timeout)
                    if retry_response.status_code == 200:
                        # Save audio file
                        with open(output_path, "wb") as f:
                            f.write(retry_response.content)
                        return output_path
                    else:
                        logger.error(f"ElevenLabs API retry error: {retry_response.status_code} {retry_response.text}")
                        return None

                logger.error(f"ElevenLabs API error: {response.status_code} {response.text}")
                return None

            # Save audio file
            with open(output_path, "wb") as f:
                f.write(response.content)

            return output_path

        except requests.exceptions.Timeout:
            logger.error(f"Timeout generating voice from ElevenLabs API")
            return None
        except Exception as e:
            logger.error(f"Error calling ElevenLabs API: {e}")
            logger.error(traceback.format_exc())
            return None

    # Maintaining async interface for compatibility
    async def process_scene(self, scene: Dict[str, Any], text: str, index: int) -> Dict[str, Any]:
        """Process a single scene, generating all necessary assets."""
        return self._process_scene_sync(scene, text, index)

    # Synchronous implementation
    def _process_scene_sync(self, scene: Dict[str, Any], text: str, index: int) -> Dict[str, Any]:
        """Synchronous implementation of scene processing."""
        try:
            # Generate images first, then voice sequentially
            logger.info(f"Processing scene {index}: Generating assets...")

            # Generate images
            image_paths = self._generate_image_sync(scene["image_prompt"], index)

            # Generate voice
            voice_path = self._generate_voice_sync(text, index)

            # Evaluate and select the best image
            if image_paths:
                candidates = self._evaluate_images_sync(image_paths, scene["image_prompt"])
                # Select the best image (highest score)
                selected_image = candidates[0].path if candidates else None
            else:
                selected_image = None

            return {
                "scene_index": index,
                "image_path": selected_image,
                "voice_path": voice_path,
                "text": text,  # Always include the spoken text for subtitles
                "text_overlay": scene.get("text_overlay"),
                "text_position": scene.get("text_position", "center"),
                "effect": scene.get("effect"),
                "transition": scene.get("transition"),
                "duration": scene.get("duration", 5),
                "all_image_candidates": image_paths,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error processing scene {index}: {e}")
            logger.error(traceback.format_exc())
            return {
                "scene_index": index,
                "error": str(e),
                "success": False
            }

    # Maintaining async interface for compatibility
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process all scenes in the visual plan."""
        return self._process_sync(state)

    def get_cache_stats_report(self) -> str:
        """Generate a report of cache statistics."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        report = (
            f"===== Semantic Cache Statistics =====\n"
            f"Hits: {self.cache_stats['hits']}\n"
            f"Misses: {self.cache_stats['misses']}\n"
            f"Hit Rate: {hit_rate:.2f}%\n"
            f"Estimated Money Saved: ${self.cache_stats['money_saved']:.2f}\n"
            f"===================================="
        )
        return report

    # Synchronous implementation
    def _process_sync(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous implementation of processing all scenes."""
        visual_plan = state["visual_plan"]
        script = state["script"]
        scenes = visual_plan["scenes"]

        # Generate hook audio if available in script
        hook_audio_path = None
        if "hook" in script:
            logger.info("Generating audio for hook...")
            hook_assets = self._generate_hook_assets(state)
            hook_audio_path = hook_assets.get("hook_audio_path")
            if hook_audio_path:
                logger.info(f"Hook audio generated: {hook_audio_path}")

        # Process all scenes sequentially
        logger.info(f"Processing {len(scenes)} scenes...")

        processed_scenes = []
        for i, scene in enumerate(scenes):
            # Get corresponding script segment
            if i < len(script["segments"]):
                segment_text = script["segments"][i]["text"]
            else:
                segment_text = f"Scene {i + 1}"

            # Process scene
            scene_result = self._process_scene_sync(scene, segment_text, i)
            time.sleep(5)
            processed_scenes.append(scene_result)

        # Check for any failures
        failures = [scene for scene in processed_scenes if not scene.get("success", True)]
        if failures:
            logger.warning(f"{len(failures)} scenes failed to process")

        # Log cache statistics
        cache_stats_report = self.get_cache_stats_report()
        logger.info(cache_stats_report)

        # Update state with processed scenes
        result = state.copy()
        result["processed_scenes"] = processed_scenes
        result["hook_audio_path"] = hook_audio_path
        result["media_success"] = len(failures) < len(processed_scenes)  # Succeed if at least some scenes processed
        result["cache_stats"] = self.cache_stats  # Include cache stats in the result

        return result


def main():
    """Test function to demonstrate the MediaGenerator functionality."""
    import sys
    import argparse
    from pprint import pprint
    import asyncio

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Test MediaGenerator functionality")
    parser.add_argument("--mode", type=str, choices=["image", "voice", "both"], default="voice",
                        help="Test mode: image, voice, or both")
    parser.add_argument("--prompt", type=str, default="A beautiful sunset over mountains with a lake in the foreground",
                        help="Image prompt to test")
    parser.add_argument("--text", type=str,
                        default="This is a test of the text to speech system. It should convert this text into spoken audio.",
                        help="Text to convert to speech")
    parser.add_argument("--async", dest="use_async", action="store_true", help="Use async interface instead of sync")
    parser.add_argument("--stability-key", type=str, help="Stability API key (or use STABILITY_API_KEY env var)")
    parser.add_argument("--elevenlabs-key", type=str, help="ElevenLabs API key (or use ELEVENLABS_API_KEY env var)")
    args = parser.parse_args()

    # Create simple configuration directly
    class DummyConfig:
        pass

    # Create the configuration structure
    config = DummyConfig()

    # Image generation configuration
    config.image_gen = DummyConfig()
    config.image_gen.provider = "stability"
    config.image_gen.api_key = args.stability_key or os.environ.get("STABILITY_API_KEY", "")
    config.image_gen.candidates_per_prompt = 1
    config.image_gen.use_image_evaluation = True

    # Text-to-speech configuration
    config.tts = DummyConfig()
    config.tts.provider = "elevenlabs"
    config.tts.api_key = args.elevenlabs_key or os.environ.get("ELEVENLABS_API_KEY", "")
    config.tts.voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "Xb7hH8MSUJpSbSDYk0k2")
    config.tts.model = "eleven_monolingual_v1"
    config.tts.voice_style = "natural"
    config.tts.speaking_rate = 1.0
    config.tts.optimize_for_platform = False

    # Platform configuration
    config.platform_config = DummyConfig()
    config.platform_config.aspect_ratio = "9:16"

    # Ensure we have API keys set up
    if args.mode in ["image", "both"] and not config.image_gen.api_key:
        print("Warning: No Stability API key found. Set STABILITY_API_KEY environment variable.")

    if args.mode in ["voice", "both"] and not config.tts.api_key:
        print("Warning: No ElevenLabs API key found. Set ELEVENLABS_API_KEY environment variable.")

    # Create MediaGenerator
    global OUTPUT_DIR
    OUTPUT_DIR = os.path.join(os.getcwd(), "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    generator = MediaGenerator(config)

    # Define async test function
    async def test_async():
        results = {}

        if args.mode in ["image", "both"]:
            print("\n---- Testing Image Generation (Async) ----")
            print(f"Prompt: '{args.prompt}'")

            try:
                image_paths = await generator.generate_image(args.prompt, 0)
                print(f"Generated {len(image_paths)} images:")
                for i, path in enumerate(image_paths):
                    print(f"  Image {i + 1}: {path}")
                    if os.path.exists(path):
                        size_kb = os.path.getsize(path) / 1024
                        print(f"    Size: {size_kb:.2f} KB")
                    else:
                        print(f"    File does not exist!")

                # Evaluate images
                if image_paths:
                    candidates = await generator.evaluate_images(image_paths, args.prompt)
                    print("\nImage Evaluation Results:")
                    for i, candidate in enumerate(candidates):
                        print(f"  Image {i + 1}: Score {candidate.score:.2f} - {candidate.path}")

                results["image_generation"] = {
                    "success": len(image_paths) > 0,
                    "count": len(image_paths),
                    "paths": image_paths
                }
            except Exception as e:
                print(f"Error testing image generation: {e}")
                traceback.print_exc()
                results["image_generation"] = {"success": False, "error": str(e)}

        if args.mode in ["voice", "both"]:
            print("\n---- Testing Voice Generation (Async) ----")
            print(f"Text: '{args.text}'")

            try:
                voice_path = await generator.generate_voice(args.text, 0)

                if voice_path and os.path.exists(voice_path):
                    size_kb = os.path.getsize(voice_path) / 1024
                    print(f"Generated voice file: {voice_path}")
                    print(f"  Size: {size_kb:.2f} KB")
                    results["voice_generation"] = {
                        "success": True,
                        "path": voice_path
                    }
                else:
                    print("Failed to generate voice file")
                    results["voice_generation"] = {
                        "success": False,
                        "path": None
                    }
            except Exception as e:
                print(f"Error testing voice generation: {e}")
                traceback.print_exc()
                results["voice_generation"] = {"success": False, "error": str(e)}

        return results

    # Define sync test function
    def test_sync():
        results = {}

        if args.mode in ["image", "both"]:
            print("\n---- Testing Image Generation (Sync) ----")
            print(f"Prompt: '{args.prompt}'")

            try:
                image_paths = generator._generate_image_sync(args.prompt, 0)
                print(f"Generated {len(image_paths)} images:")
                for i, path in enumerate(image_paths):
                    print(f"  Image {i + 1}: {path}")
                    if os.path.exists(path):
                        size_kb = os.path.getsize(path) / 1024
                        print(f"    Size: {size_kb:.2f} KB")
                    else:
                        print(f"    File does not exist!")

                # Evaluate images
                if image_paths:
                    candidates = generator._evaluate_images_sync(image_paths, args.prompt)
                    print("\nImage Evaluation Results:")
                    for i, candidate in enumerate(candidates):
                        print(f"  Image {i + 1}: Score {candidate.score:.2f} - {candidate.path}")

                results["image_generation"] = {
                    "success": len(image_paths) > 0,
                    "count": len(image_paths),
                    "paths": image_paths
                }
            except Exception as e:
                print(f"Error testing image generation: {e}")
                traceback.print_exc()
                results["image_generation"] = {"success": False, "error": str(e)}

        if args.mode in ["voice", "both"]:
            print("\n---- Testing Voice Generation (Sync) ----")
            print(f"Text: '{args.text}'")

            try:
                voice_path = generator._generate_voice_sync(args.text, 0)

                if voice_path and os.path.exists(voice_path):
                    size_kb = os.path.getsize(voice_path) / 1024
                    print(f"Generated voice file: {voice_path}")
                    print(f"  Size: {size_kb:.2f} KB")
                    results["voice_generation"] = {
                        "success": True,
                        "path": voice_path
                    }
                else:
                    print("Failed to generate voice file")
                    results["voice_generation"] = {
                        "success": False,
                        "path": None
                    }
            except Exception as e:
                print(f"Error testing voice generation: {e}")
                traceback.print_exc()
                results["voice_generation"] = {"success": False, "error": str(e)}

        return results

    # Run appropriate test based on mode
    if args.use_async:
        print("Using asynchronous interface for testing")
        results = asyncio.run(test_async())
    else:
        print("Using synchronous interface for testing")
        results = test_sync()

    # Print summary
    print("\n---- Test Summary ----")
    success = all(r.get("success", False) for r in results.values()) if results else False
    print(f"Overall test {'succeeded' if success else 'failed'}")

    return results


if __name__ == "__main__":
    main()