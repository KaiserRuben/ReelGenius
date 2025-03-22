from typing import Dict, List, Any, Optional, Union
import os
import tempfile
import json
import time
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage
from loguru import logger
import subprocess

# Correct MoviePy imports
from moviepy.video.VideoClip import ImageClip, TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip, concatenate_videoclips
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import CompositeAudioClip, concatenate_audioclips
from moviepy.video.io.VideoFileClip import VideoFileClip

# Import effects individually
from moviepy.video.fx.FadeIn import FadeIn
from moviepy.video.fx.FadeOut import FadeOut
from moviepy.video.fx.CrossFadeIn import CrossFadeIn
from moviepy.video.fx.CrossFadeOut import CrossFadeOut
from moviepy.video.fx.MultiplyColor import MultiplyColor
from PIL import Image

from ..config import PipelineConfig, OUTPUT_DIR
from ..models.llm import DeepSeekChatModel, PromptTemplateManager


# Define a namespace for video and audio effects for easier transition
class vfx:
    FadeIn = FadeIn
    FadeOut = FadeOut
    CrossFadeIn = CrossFadeIn
    CrossFadeOut = CrossFadeOut

    @staticmethod
    def MultiplySpeed(clip, factor):
        """Apply speed changes to the clip"""
        return clip.fx(lambda c: c.set_duration(c.duration / factor))


class afx:
    @staticmethod
    def AudioFadeIn(clip, duration):
        """Create a audio fade in effect"""
        return clip.volumex(lambda t: min(t / duration, 1) if duration > 0 else 1)

    @staticmethod
    def AudioFadeOut(clip, duration):
        """Create a audio fade out effect"""
        return clip.volumex(lambda t: max(0, 1 - t / duration) if duration > 0 and t <= clip.duration else 1)


class VideoAssembler:
    """Assemble final video from media assets."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = os.path.join(OUTPUT_DIR, "videos")
        os.makedirs(self.output_dir, exist_ok=True)

        # Font settings based on platform
        self.font = "Helvetica"
        self.fontsize = 40
        self.text_color = 'white'
        self.text_bg_color = (0, 0, 0, 170)  # Black with alpha

        # Initialize LLM for metadata generation
        self.llm = DeepSeekChatModel(
            api_key=config.llm.api_key,
            model=config.llm.model,
            base_url=config.llm.base_url,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            timeout=config.llm.timeout
        )
        self.prompt_manager = PromptTemplateManager()

    def _create_text_overlay(self, text: str, position: str, duration: int, size) -> TextClip:
        """Create a text overlay for a scene."""
        # Limit text length for overlay
        if len(text) > 100:
            text = text[:97] + "..."

        # Create text clip with background
        txt_clip = TextClip(
            text,
            fontsize=self.fontsize,
            font=self.font,
            color=self.text_color,
            bg_color=self.text_bg_color,
            method='caption',
            align='center',
            size=(size[0] - 40, None)  # Width slightly less than image width
        )

        # Position the text
        if position == "top":
            txt_clip = txt_clip.set_position(('center', 20))
        elif position == "bottom":
            txt_clip = txt_clip.set_position(('center', 'bottom'))
        else:  # center or default
            txt_clip = txt_clip.set_position('center')

        # Set duration and add fade
        txt_clip = txt_clip.set_duration(duration)
        # Corrected fade effects
        txt_clip = FadeIn(txt_clip, 0.5)
        txt_clip = FadeOut(txt_clip, 0.5)

        return txt_clip

    def _apply_transition(self, clip1, clip2, transition_type: str, duration: float = 0.5):
        """Apply transition between two clips."""
        if transition_type == "fade" or transition_type == "crossfade":
            # Create a crossfade transition
            clip1 = FadeOut(clip1, duration)
            clip2 = FadeIn(clip2, duration)

            transition_clip = concatenate_videoclips(
                [clip1.subclip(0, clip1.duration - duration),
                 clip2],
                method="compose"
            )

            return transition_clip

        elif transition_type == "wipe":
            # Implement a simple wipe transition
            def make_frame(t):
                if t < duration:
                    # Calculate progress ratio
                    ratio = t / duration
                    frame1 = clip1.get_frame(clip1.duration - duration + t)
                    frame2 = clip2.get_frame(t)
                    width = frame1.shape[1]
                    # Create wipe effect from left to right
                    split = int(width * ratio)
                    frame = frame1.copy()
                    frame[:, split:] = frame2[:, split:]
                    return frame
                else:
                    return clip2.get_frame(t)

            from moviepy.video.VideoClip import VideoClip
            transition = VideoClip(make_frame=make_frame, duration=duration)
            transition = transition.set_fps(clip1.fps)

            if clip1.audio is not None and clip2.audio is not None:
                # Cross-fade audio
                audio1 = clip1.audio.subclip(clip1.duration - duration, clip1.duration)
                audio2 = clip2.audio.subclip(0, duration)

                audio1 = audio1.volumex(lambda t: 1 - t / duration)
                audio2 = audio2.volumex(lambda t: t / duration)

                transition_audio = CompositeAudioClip([audio1, audio2])
                transition = transition.set_audio(transition_audio)

            # Combine clips with transition
            return concatenate_videoclips([
                clip1.subclip(0, clip1.duration - duration),
                transition,
                clip2.subclip(duration)
            ])

        else:
            # Default to cut (no transition)
            return concatenate_videoclips([clip1, clip2])

    def assemble_video(self, media_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble final video from processed scenes."""
        processed_scenes = media_data["processed_scenes"]
        script_data = media_data["script_data"]
        script = script_data["script"]

        # Create a unique output filename
        output_filename = f"video_{int(time.time())}.mp4"
        output_path = os.path.join(self.output_dir, output_filename)

        try:
            # Filter out scenes with missing assets
            valid_scenes = [
                scene for scene in processed_scenes
                if "image_path" in scene and scene["image_path"] and os.path.exists(scene["image_path"]) and
                   "voice_path" in scene and scene["voice_path"] and os.path.exists(scene["voice_path"])
            ]

            if not valid_scenes:
                logger.error("No valid scenes to assemble")
                return {
                    "success": False,
                    "error": "No valid scenes to assemble",
                    "media_data": media_data
                }

            # Process scenes to create video clips
            logger.info(f"Assembling {len(valid_scenes)} scenes into final video...")
            video_clips = []

            # Parse aspect ratio to determine dimensions
            aspect_ratio = self.config.platform_config.aspect_ratio
            if aspect_ratio == "9:16":
                target_width, target_height = 1080, 1920
            elif aspect_ratio == "16:9":
                target_width, target_height = 1920, 1080
            else:
                # Parse custom ratio
                try:
                    width_ratio, height_ratio = map(int, aspect_ratio.split(":"))
                    if width_ratio > height_ratio:
                        target_width, target_height = 1920, int(1920 * height_ratio / width_ratio)
                    else:
                        target_height, target_width = 1920, int(1920 * width_ratio / height_ratio)
                except:
                    # Default to 16:9
                    target_width, target_height = 1920, 1080

            for i, scene in enumerate(valid_scenes):
                try:
                    # Load image
                    img_clip = ImageClip(scene["image_path"])

                    # Resize image to target dimensions while maintaining aspect ratio
                    img_width, img_height = img_clip.size

                    # Apply Ken Burns effect (subtle zoom) if motion effects enabled
                    if self.config.visual.motion_effects:
                        # Zoom in slightly over the duration
                        zoom_factor = 1.05
                        img_clip = img_clip.resize(lambda t: 1 + (zoom_factor - 1) * t / scene["duration"])

                    # Resize to target dimensions
                    img_clip = img_clip.resize(width=target_width, height=target_height)

                    # Set duration
                    img_clip = img_clip.set_duration(scene["duration"])

                    # Add text overlay if specified
                    if scene.get("text_overlay") and self.config.platform_config.text_overlay:
                        txt_clip = self._create_text_overlay(
                            scene["text_overlay"],
                            scene["text_position"],
                            scene["duration"],
                            (target_width, target_height)
                        )

                        # Apply text animation if enabled
                        if self.config.visual.text_animation:
                            txt_clip = FadeIn(txt_clip, 0.5)
                            txt_clip = FadeOut(txt_clip, 0.5)

                        # Combine image and text
                        video_clip = CompositeVideoClip([img_clip, txt_clip])
                    else:
                        video_clip = img_clip

                    # Add audio
                    audio_clip = AudioFileClip(scene["voice_path"])

                    # If audio is longer than clip duration, extend clip duration
                    if audio_clip.duration > video_clip.duration:
                        video_clip = video_clip.set_duration(audio_clip.duration)
                    # If audio is shorter, extend audio with silence
                    elif audio_clip.duration < video_clip.duration:
                        from pydub import AudioSegment
                        silence_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
                        silence = AudioSegment.silent(duration=(video_clip.duration - audio_clip.duration) * 1000)
                        silence.export(silence_path, format="mp3")

                        # Concatenate audio
                        silence_clip = AudioFileClip(silence_path)
                        audio_clip = concatenate_audioclips([audio_clip, silence_clip])

                    # Set audio
                    video_clip = video_clip.set_audio(audio_clip)

                    # Add to list
                    video_clips.append(video_clip)

                except Exception as e:
                    logger.error(f"Error processing scene {i}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

            if not video_clips:
                logger.error("No valid video clips to assemble")
                return {
                    "success": False,
                    "error": "Failed to create any valid video clips",
                    "media_data": media_data
                }

            # Apply transitions between clips
            logger.info("Applying transitions between scenes...")
            final_clips = []

            if len(video_clips) == 1:
                final_clips = video_clips
            else:
                # Create a list of transitions from scene data
                transitions = [scene.get("transition", "fade") for scene in valid_scenes[:-1]]

                # Apply the first clip
                result_clip = video_clips[0]

                # Apply transitions sequentially
                for i in range(len(video_clips) - 1):
                    # Get transition type
                    transition_type = transitions[i]

                    # Apply transition to next clip
                    result_clip = self._apply_transition(
                        result_clip,
                        video_clips[i + 1],
                        transition_type,
                        duration=self.config.visual.transition_style == "smooth" and 0.75 or 0.5
                    )

                final_clips = [result_clip]

            # Concatenate all clips if we have multiple
            logger.info("Finalizing video...")
            if len(final_clips) > 1:
                final_video = concatenate_videoclips(final_clips)
            else:
                final_video = final_clips[0]

            # Add background music if available
            # NOTE: This would be implemented with a music library

            # Add hook overlay at the beginning if specified
            if "hook" in script and self.config.platform_config.optimize_hook:
                hook_duration = self.config.platform_config.hook_duration
                hook_font_size = self.fontsize + 10  # Bigger font for hook

                hook_txt = TextClip(
                    script["hook"],
                    fontsize=hook_font_size,
                    font=self.font,
                    color=self.text_color,
                    bg_color=self.text_bg_color,
                    method='caption',
                    align='center',
                    size=(target_width - 40, None)
                )
                hook_txt = hook_txt.set_position('center').set_duration(hook_duration)
                hook_txt = FadeIn(hook_txt, 0.5)
                hook_txt = FadeOut(hook_txt, 0.5)

                # Overlay hook on beginning of video
                hook_section = final_video.subclip(0, min(hook_duration, final_video.duration))
                hook_overlay = CompositeVideoClip([hook_section, hook_txt])

                # Replace beginning with hook overlay
                if final_video.duration > hook_duration:
                    final_video = concatenate_videoclips([
                        hook_overlay,
                        final_video.subclip(hook_duration)
                    ])
                else:
                    final_video = hook_overlay

            # Apply visual filters if enabled
            if self.config.visual.apply_filters:
                logger.info("Applying visual filters...")
                # Apply a subtle color grading based on the color scheme
                if self.config.visual.color_scheme == "vibrant":
                    # Increase saturation slightly
                    final_video = MultiplyColor(final_video, 1.2)
                elif self.config.visual.color_scheme == "muted":
                    # Decrease saturation slightly
                    final_video = MultiplyColor(final_video, 0.8)
            # Write final video
            logger.info(f"Writing video to {output_path}...")
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=tempfile.NamedTemporaryFile(suffix='.m4a').name,
                remove_temp=True,
                threads=4,
                fps=self.config.platform_config.framerate
            )

            logger.info(f"Video assembled successfully: {output_path}")

            # Return results
            return {
                "success": True,
                "video_path": output_path,
                "duration": final_video.duration,
                "resolution": f"{target_width}x{target_height}",
                "media_data": media_data
            }

        except Exception as e:
            logger.error(f"Error assembling video: {e}")
            import traceback
            logger.error(traceback.format_exc())

            return {
                "success": False,
                "error": str(e),
                "media_data": media_data
            }

    def generate_metadata(self, script_data: Dict[str, Any], assembly_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate platform-optimized metadata for the video."""
        # Get prompt template
        prompt_data = self.config.get_prompts_data()
        prompt_data.update({
            "script": script_data["script"],
            "content_strategy": script_data["processed_input"]["content_strategy"]
        })

        prompt = self.prompt_manager.render_template("metadata_generation", **prompt_data)

        # Use meta-prompting if enabled
        if self.config.llm.use_meta_prompting:
            prompt = self.prompt_manager.optimize_prompt(
                prompt,
                "metadata generation",
                self.config.model_dump()
            )

        output_schema = {
            "title": "string",
            "description": "string",
            "hashtags": ["string"],
            "category": "string"
        }

        messages = [
            SystemMessage(content=f"You are a {self.config.platform} metadata optimization expert."),
            HumanMessage(content=prompt)
        ]

        try:
            # Log the prompt if detailed logging is enabled
            if self.config.enable_detailed_logging:
                logger.info(f"Metadata Generation Prompt:\n{prompt}")

            result = self.llm.structured_output(messages, output_schema)

            # Process hashtags
            hashtags = result.get("hashtags", [])
            # Ensure all hashtags start with #
            hashtags = [tag if tag.startswith("#") else f"#{tag}" for tag in hashtags]

            metadata = {
                "title": result.get("title", f"Video {int(time.time())}"),
                "description": result.get("description", ""),
                "hashtags": hashtags,
                "category": result.get("category", "Entertainment"),
                "platform": self.config.platform,
                "video_path": assembly_result.get("video_path", ""),
                "duration": assembly_result.get("duration", 0),
                "resolution": assembly_result.get("resolution", "")
            }

            # Save metadata
            if "video_path" in assembly_result:
                metadata_path = os.path.join(
                    OUTPUT_DIR,
                    "metadata",
                    f"{os.path.basename(assembly_result['video_path']).split('.')[0]}.json"
                )

                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                metadata["metadata_path"] = metadata_path

            logger.info(f"Generated metadata with title: {metadata['title']}")
            return metadata

        except Exception as e:
            logger.error(f"Error generating metadata: {e}")

            # Create basic metadata
            basic_metadata = {
                "title": f"Video {int(time.time())}",
                "description": script_data["script"].get("hook", ""),
                "hashtags": [f"#{self.config.platform}"],
                "category": "Entertainment",
                "platform": self.config.platform,
                "video_path": assembly_result.get("video_path", ""),
                "duration": assembly_result.get("duration", 0)
            }

            return basic_metadata

    def process(self, media_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble video and generate metadata."""
        # Assemble video
        logger.info("Assembling final video...")
        assembly_result = self.assemble_video(media_data)

        if not assembly_result["success"]:
            logger.error(f"Video assembly failed: {assembly_result.get('error', 'Unknown error')}")
            return assembly_result

        # Generate metadata
        logger.info("Generating video metadata...")
        script_data = media_data["script_data"]
        metadata = self.generate_metadata(script_data, assembly_result)

        return {
            "success": True,
            "video_path": assembly_result["video_path"],
            "metadata": metadata,
            "assembly_result": assembly_result
        }
