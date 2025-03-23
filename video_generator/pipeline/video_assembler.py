from typing import Dict, List, Any, Optional, Union
import os
import tempfile
import json
import time
from pathlib import Path

import moviepy
from langchain_core.messages import SystemMessage, HumanMessage
from loguru import logger
import traceback

from moviepy import TextClip, concatenate_videoclips, CompositeAudioClip, ImageClip, CompositeVideoClip, AudioFileClip, \
    concatenate_audioclips

# Check for MoviePy dependencies
MOVIEPY_AVAILABLE = True
try:
    from moviepy.video.fx.fadeout import fadeout
    from moviepy.video.fx.fadein import fadein

    FadeIn = fadein
    FadeOut = fadeout
except ImportError:
    try:
        from moviepy.video.fx.FadeIn import FadeIn
        from moviepy.video.fx.FadeOut import FadeOut
    except ImportError:
        logger.warning("Could not import FadeIn/FadeOut effects. Using basic transitions.")


        # Define simple alternatives
        def FadeIn(clip, duration=1.0):
            return clip.fx(lambda t: min(1, t / duration) if t < duration else 1)


        def FadeOut(clip, duration=1.0):
            return clip.fx(lambda gf, t: gf(t) * (1 - min(1, t / duration))
            if t > clip.duration - duration else gf(t))

try:
    # Try importing other effects that might be needed
    from moviepy.video.fx.crossfadein import crossfadein as CrossFadeIn
    from moviepy.video.fx.crossfadeout import crossfadeout as CrossFadeOut
except ImportError:
    try:
        from moviepy.video.fx.CrossFadeIn import CrossFadeIn
        from moviepy.video.fx.CrossFadeOut import CrossFadeOut
    except ImportError:
        logger.warning("Could not import CrossFadeIn/CrossFadeOut effects. Using basic transitions.")
        # Define simple alternatives
        CrossFadeIn = FadeIn
        CrossFadeOut = FadeOut

# Conditionally import and check for PIL
PIL_AVAILABLE = True
try:
    from PIL import Image, ImageFilter
except ImportError:
    logger.error("PIL (Pillow) not available. Image processing will be limited.")
    PIL_AVAILABLE = False

from ..config import PipelineConfig, OUTPUT_DIR
from ..models.llm import DeepSeekChatModel, PromptTemplateManager


# Define a namespace for video and audio effects for easier transition and compatibility
class vfx:
    @staticmethod
    def FadeIn(clip, duration=1.0):
        """Apply fade in effect with compatibility handling."""
        if hasattr(clip, 'fadein'):
            return clip.fadein(duration)
        elif 'FadeIn' in dir(moviepy.video.fx):
            return moviepy.video.fx.FadeIn(clip, duration)
        else:
            # Manual implementation as fallback
            return clip.fx(lambda t: min(1, t / duration) if t < duration else 1)

    @staticmethod
    def FadeOut(clip, duration=1.0):
        """Apply fade out effect with compatibility handling."""
        if hasattr(clip, 'fadeout'):
            return clip.fadeout(duration)
        elif 'FadeOut' in dir(moviepy.video.fx):
            return moviepy.video.fx.FadeOut(clip, duration)
        else:
            # Manual implementation as fallback
            def fadeout_func(get_frame, t):
                if t > clip.duration - duration:
                    return get_frame(t) * (1 - (t - (clip.duration - duration)) / duration)
                else:
                    return get_frame(t)

            return clip.fx(fadeout_func)

    @staticmethod
    def CrossFadeIn(clip, duration=1.0):
        """Apply cross-fade in effect with compatibility handling."""
        try:
            if 'crossfadein' in dir(moviepy.video.fx):
                return moviepy.video.fx.crossfadein(clip, duration)
            elif 'CrossFadeIn' in dir(moviepy.video.fx):
                return moviepy.video.fx.CrossFadeIn(clip, duration)
            else:
                # Fall back to regular fade in if cross fade not available
                return vfx.FadeIn(clip, duration)
        except Exception as e:
            logger.error(f"Error applying CrossFadeIn: {e}")
            return clip  # Return original clip on error

    @staticmethod
    def CrossFadeOut(clip, duration=1.0):
        """Apply cross fade out effect with compatibility handling."""
        try:
            if 'crossfadeout' in dir(moviepy.video.fx):
                return moviepy.video.fx.crossfadeout(clip, duration)
            elif 'CrossFadeOut' in dir(moviepy.video.fx):
                return moviepy.video.fx.CrossFadeOut(clip, duration)
            else:
                # Fall back to regular fade out if cross fade not available
                return vfx.FadeOut(clip, duration)
        except Exception as e:
            logger.error(f"Error applying CrossFadeOut: {e}")
            return clip  # Return original clip on error

    @staticmethod
    def MultiplySpeed(clip, factor):
        """Apply speed changes to the clip"""
        if hasattr(clip, 'speedx'):
            return clip.speedx(factor)
        else:
            return clip.fx(lambda c: c.set_duration(c.duration / factor))


class afx:
    @staticmethod
    def AudioFadeIn(clip, duration):
        """Create a audio fade in effect"""
        if hasattr(clip, 'audio_fadein'):
            return clip.audio_fadein(duration)
        else:
            return clip.volumex(lambda t: min(t / duration, 1) if duration > 0 else 1)

    @staticmethod
    def AudioFadeOut(clip, duration):
        """Create a audio fade out effect"""
        if hasattr(clip, 'audio_fadeout'):
            return clip.audio_fadeout(duration)
        else:
            return clip.volumex(lambda t: max(0, 1 - t / duration) if duration > 0 and t <= clip.duration else 1)


class VideoAssembler:
    """Assemble final video from media assets."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = os.path.join(OUTPUT_DIR, "videos")
        os.makedirs(self.output_dir, exist_ok=True)

        # Check if MoviePy is available
        if not MOVIEPY_AVAILABLE:
            logger.error("MoviePy is not available. Video assembly will not work.")
            self.moviepy_available = False
        else:
            self.moviepy_available = True

        # Font settings based on platform
        self.font = "Helvetica"  # Default font
        self.fontsize = 40
        self.text_color = 'white'
        self.text_bg_color = (0, 0, 0, 170)  # Black with alpha

        # Try to find a suitable font if Helvetica is not available
        try:
            from moviepy.video.tools.drawing import findFont
            available_font = findFont('Helvetica')
            if available_font:
                self.font = available_font
            else:
                # Try other common fonts
                for font_name in ['Arial', 'DejaVuSans', 'FreeSans', 'Liberation']:
                    available_font = findFont(font_name)
                    if available_font:
                        self.font = available_font
                        break
        except:
            # If finding fonts fails, we'll use the default
            pass

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

    def _create_text_overlay(self, text: str, position: str, duration: int, size) -> Any:
        """Create a text overlay for a scene."""
        if not self.moviepy_available:
            logger.error("MoviePy is not available. Cannot create text overlay.")
            return None

        try:
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
            # Use our compatibility wrappers for effects
            txt_clip = vfx.FadeIn(txt_clip, 0.5)
            txt_clip = vfx.FadeOut(txt_clip, 0.5)

            return txt_clip
        except Exception as e:
            logger.error(f"Error creating text overlay: {e}")
            logger.error(traceback.format_exc())
            return None

    def _apply_transition(self, clip1, clip2, transition_type: str, duration: float = 0.5):
        """Apply transition between two clips with improved compatibility."""
        if not self.moviepy_available:
            logger.error("MoviePy is not available. Cannot apply transitions.")
            return None

        try:
            if transition_type == "fade" or transition_type == "crossfade":
                # Create a crossfade transition
                clip1 = vfx.FadeOut(clip1, duration)
                clip2 = vfx.FadeIn(clip2, duration)

                # Create crossfade effect - handle different MoviePy versions
                try:
                    transition_clip = concatenate_videoclips(
                        [clip1.subclip(0, clip1.duration - duration),
                         clip2],
                        method="compose"
                    )
                except TypeError:
                    # For older versions that may not support "compose"
                    transition_clip = concatenate_videoclips(
                        [clip1.subclip(0, clip1.duration - duration),
                         clip2]
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

                # Handle audio transition
                if hasattr(clip1, 'audio') and clip1.audio is not None and hasattr(clip2,
                                                                                   'audio') and clip2.audio is not None:
                    # Cross-fade audio
                    audio1 = clip1.audio.subclip(clip1.duration - duration, clip1.duration)
                    audio2 = clip2.audio.subclip(0, duration)

                    # Apply volume adjustments with compatibility checks
                    if hasattr(audio1, 'volumex'):
                        audio1 = audio1.volumex(lambda t: 1 - t / duration)
                        audio2 = audio2.volumex(lambda t: t / duration)
                    else:
                        # Fallback method if volumex is not available
                        audio1 = audio1.fl(lambda gf, t: gf(t) * (1 - t / duration))
                        audio2 = audio2.fl(lambda gf, t: gf(t) * (t / duration))

                    try:
                        transition_audio = CompositeAudioClip([audio1, audio2])
                        transition = transition.set_audio(transition_audio)
                    except:
                        logger.warning("Could not create composite audio for transition. Using silent transition.")

                # Combine clips with transition
                try:
                    return concatenate_videoclips([
                        clip1.subclip(0, clip1.duration - duration),
                        transition,
                        clip2.subclip(duration)
                    ])
                except:
                    logger.warning("Error in wipe transition. Falling back to simple cut.")
                    return concatenate_videoclips([clip1, clip2])

            else:
                # Default to cut (no transition)
                return concatenate_videoclips([clip1, clip2])

        except Exception as e:
            logger.error(f"Error applying transition: {e}")
            logger.error(traceback.format_exc())
            # Fall back to simple concatenation
            try:
                return concatenate_videoclips([clip1, clip2])
            except:
                logger.error("Failed to concatenate clips even with simple method")
                return clip1  # Just return the first clip as a last resort

    def assemble_video(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble final video from processed scenes."""
        if not self.moviepy_available:
            logger.error("MoviePy is not available. Cannot assemble video.")
            return {
                "success": False,
                "error": "MoviePy library is not available. Cannot assemble video.",
                "state": state
            }

        processed_scenes = state["processed_scenes"]
        script = state["script"]

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
                    "state": state
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
                        try:
                            # Zoom in slightly over the duration
                            zoom_factor = 1.05
                            img_clip = img_clip.resize(lambda t: 1 + (zoom_factor - 1) * t / scene["duration"])
                        except:
                            logger.warning("Could not apply zoom effect. Using static image.")

                    # Resize to target dimensions
                    try:
                        img_clip = img_clip.resize(width=target_width, height=target_height)
                    except:
                        logger.warning(f"Error resizing image to {target_width}x{target_height}. Using original size.")

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

                        if txt_clip:
                            # Apply text animation if enabled
                            if self.config.visual.text_animation:
                                try:
                                    txt_clip = vfx.FadeIn(txt_clip, 0.5)
                                    txt_clip = vfx.FadeOut(txt_clip, 0.5)
                                except:
                                    logger.warning("Could not apply text animation. Using static text.")

                            # Combine image and text
                            try:
                                video_clip = CompositeVideoClip([img_clip, txt_clip])
                            except:
                                logger.warning("Error creating composite with text overlay. Using image only.")
                                video_clip = img_clip
                        else:
                            video_clip = img_clip
                    else:
                        video_clip = img_clip

                    # Add audio
                    try:
                        audio_clip = AudioFileClip(scene["voice_path"])

                        # If audio is longer than clip duration, extend clip duration
                        if audio_clip.duration > video_clip.duration:
                            video_clip = video_clip.set_duration(audio_clip.duration)
                        # If audio is shorter, extend audio with silence
                        elif audio_clip.duration < video_clip.duration:
                            try:
                                from pydub import AudioSegment
                                silence_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
                                silence = AudioSegment.silent(
                                    duration=(video_clip.duration - audio_clip.duration) * 1000)
                                silence.export(silence_path, format="mp3")

                                # Concatenate audio
                                silence_clip = AudioFileClip(silence_path)
                                audio_clip = concatenate_audioclips([audio_clip, silence_clip])
                            except ImportError:
                                logger.warning("pydub not available. Audio might end before video.")
                            except:
                                logger.warning("Error extending audio with silence. Audio might end before video.")

                        # Set audio
                        video_clip = video_clip.set_audio(audio_clip)
                    except:
                        logger.warning(f"Error adding audio to scene {i}. Video will be silent.")

                    # Add to list
                    video_clips.append(video_clip)

                except Exception as e:
                    logger.error(f"Error processing scene {i}: {e}")
                    logger.error(traceback.format_exc())

            if not video_clips:
                logger.error("No valid video clips to assemble")
                return {
                    "success": False,
                    "error": "Failed to create any valid video clips",
                    "state": state
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
                    try:
                        # Get transition type
                        transition_type = transitions[i]

                        # Apply transition to next clip
                        result_clip = self._apply_transition(
                            result_clip,
                            video_clips[i + 1],
                            transition_type,
                            duration=self.config.visual.transition_style == "smooth" and 0.75 or 0.5
                        )
                    except Exception as e:
                        logger.error(f"Error applying transition {i}: {e}")
                        logger.error(traceback.format_exc())
                        # Try simple concatenation as fallback
                        try:
                            result_clip = concatenate_videoclips([result_clip, video_clips[i + 1]])
                        except:
                            logger.error(f"Failed to concatenate clips {i} and {i + 1}")
                            # Just keep what we have so far
                            break

                final_clips = [result_clip]

            # Concatenate all clips if we have multiple
            logger.info("Finalizing video...")
            if len(final_clips) > 1:
                try:
                    final_video = concatenate_videoclips(final_clips)
                except Exception as e:
                    logger.error(f"Error concatenating final clips: {e}")
                    logger.error(traceback.format_exc())
                    # Use the first clip as fallback
                    final_video = final_clips[0]
            else:
                final_video = final_clips[0]

            # Add background music if available
            # NOTE: This would be implemented with a music library

            # Add hook overlay at the beginning if specified
            if "hook" in script and self.config.platform_config.optimize_hook:
                try:
                    hook_duration = min(self.config.platform_config.hook_duration, final_video.duration)
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
                    hook_txt = vfx.FadeIn(hook_txt, 0.5)
                    hook_txt = vfx.FadeOut(hook_txt, 0.5)

                    # Overlay hook on beginning of video
                    hook_section = final_video.subclip(0, hook_duration)
                    hook_overlay = CompositeVideoClip([hook_section, hook_txt])

                    # Replace beginning with hook overlay
                    if final_video.duration > hook_duration:
                        final_video = concatenate_videoclips([
                            hook_overlay,
                            final_video.subclip(hook_duration)
                        ])
                    else:
                        final_video = hook_overlay
                except Exception as e:
                    logger.error(f"Error adding hook overlay: {e}")
                    logger.error(traceback.format_exc())

            # Apply visual filters if enabled
            if self.config.visual.apply_filters:
                try:
                    logger.info("Applying visual filters...")

                    # Check if MoviePy's MultiplyColor effect is available
                    multiply_available = False
                    try:
                        from moviepy.video.fx.colorx import colorx
                        multiply_available = True
                    except ImportError:
                        try:
                            from moviepy.video.fx.MultiplyColor import MultiplyColor
                            multiply_available = True
                        except ImportError:
                            logger.warning("Could not import color effects. Using basic filters.")

                    # Apply a subtle color grading based on the color scheme
                    if multiply_available:
                        if self.config.visual.color_scheme == "vibrant":
                            # Increase saturation slightly
                            try:
                                final_video = colorx(final_video, 1.2)
                            except:
                                try:
                                    from moviepy.video.fx.MultiplyColor import MultiplyColor
                                    final_video = MultiplyColor(final_video, 1.2)
                                except:
                                    logger.warning("Could not apply vibrant filter.")

                        elif self.config.visual.color_scheme == "muted":
                            # Decrease saturation slightly
                            try:
                                final_video = colorx(final_video, 0.8)
                            except:
                                try:
                                    from moviepy.video.fx.MultiplyColor import MultiplyColor
                                    final_video = MultiplyColor(final_video, 0.8)
                                except:
                                    logger.warning("Could not apply muted filter.")
                except Exception as e:
                    logger.error(f"Error applying visual filters: {e}")

            # Write final video
            logger.info(f"Writing video to {output_path}...")

            # Create temporary directory for intermediate files
            temp_dir = tempfile.mkdtemp()
            temp_audio = os.path.join(temp_dir, 'temp_audio.m4a')

            try:
                # Use a safer set of parameters for write_videofile
                final_video.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile=temp_audio,
                    remove_temp=True,
                    threads=4,
                    fps=self.config.platform_config.framerate,
                    preset='medium',  # Less demanding preset
                    bitrate='5000k'  # Reasonable bitrate
                )
            except Exception as e:
                logger.error(f"Error writing video file: {e}")
                logger.error(traceback.format_exc())

                # Try with simpler parameters as fallback
                try:
                    logger.info("Retrying with simpler video export settings...")
                    final_video.write_videofile(
                        output_path,
                        codec='libx264',
                        audio_codec='aac',
                        threads=2,
                        fps=24,
                        preset='ultrafast'  # Less quality but more reliable
                    )
                except Exception as e2:
                    logger.error(f"Second attempt at writing video also failed: {e2}")
                    return {
                        "success": False,
                        "error": f"Failed to write video file: {str(e2)}",
                        "media_data": media_data
                    }

            # Cleanup temporary directory
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass

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
            logger.error(traceback.format_exc())

            return {
                "success": False,
                "error": str(e),
                "state": state
            }

    def generate_metadata(self, state: Dict[str, Any], assembly_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate platform-optimized metadata for the video."""
        # Get prompt template
        prompt_data = self.config.get_prompts_data()
        prompt_data.update({
            "script": state["script"],
            "content_strategy": state["content_strategy"]
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
                metadata_dir = Path(OUTPUT_DIR) / "metadata"
                os.makedirs(metadata_dir, exist_ok=True)

                metadata_path = os.path.join(
                    metadata_dir,
                    f"{os.path.basename(assembly_result['video_path']).split('.')[0]}.json"
                )

                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                metadata["metadata_path"] = metadata_path

            logger.info(f"Generated metadata with title: {metadata['title']}")
            return metadata

        except Exception as e:
            logger.error(f"Error generating metadata: {e}")
            logger.error(traceback.format_exc())

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

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble video and generate metadata."""
        # Check if MoviePy is available
        if not self.moviepy_available:
            logger.error("MoviePy is not available. Video assembly will not work.")
            result = state.copy()
            result["success"] = False
            result["error"] = "MoviePy library is not available. Cannot assemble video."
            return result

        # Assemble video
        logger.info("Assembling final video...")
        assembly_result = self.assemble_video(state)

        if not assembly_result["success"]:
            logger.error(f"Video assembly failed: {assembly_result.get('error', 'Unknown error')}")
            result = state.copy()
            result["success"] = False
            result["error"] = assembly_result.get("error", "Unknown error")
            return result

        # Generate metadata
        logger.info("Generating video metadata...")
        metadata = self.generate_metadata(state, assembly_result)

        # Update state with video path and metadata
        result = state.copy()
        result["success"] = True
        result["video_path"] = assembly_result["video_path"]
        result["metadata"] = metadata
        
        return result
