from typing import Dict, List, Any, Optional, Union, Tuple
import os
import tempfile
import json
import time
import math
import random
from pathlib import Path
import traceback

from moviepy.audio.fx import AudioFadeOut, AudioFadeIn
# Import proper MoviePy modules as per v2.1.2 (avoid moviepy.editor)
from moviepy.video.VideoClip import TextClip, ImageClip, ColorClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy import concatenate_videoclips
from moviepy.audio.AudioClip import CompositeAudioClip, concatenate_audioclips
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.fx import FadeIn, FadeOut
from moviepy.video.io.VideoFileClip import VideoFileClip
from langchain_core.messages import SystemMessage, HumanMessage
from loguru import logger

# Try to import optional dependencies
try:
    from pydub import AudioSegment

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning("pydub not available. Audio normalization will be limited.")

try:
    from PIL import Image, ImageFilter, ImageEnhance

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL (Pillow) not available. Image processing will be limited.")

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available. Some effects will be limited.")

from ..config import PipelineConfig, OUTPUT_DIR
from ..models.llm import DeepSeekChatModel, PromptTemplateManager


# Define a namespace for video and audio effects with compatibility layers
class VideoEffects:
    """Modern video effects compatible with MoviePy 2.1.2"""

    @staticmethod
    def fade_in(clip, duration=1.0):
        """Apply fade in effect with compatibility handling."""
        return clip.with_effects([FadeIn(duration)])

    @staticmethod
    def fade_out(clip, duration=1.0):
        """Apply fade out effect with compatibility handling."""
        return clip.with_effects([FadeOut(duration)])

    @staticmethod
    def zoom_in(clip, duration=None, zoom_to=1.2):
        """Apply zoom in effect"""
        if duration is None:
            duration = clip.duration / 2

        def zoom_func(t):
            progress = min(1, t / duration)
            zoom_factor = 1 + (zoom_to - 1) * progress
            return zoom_factor

        return clip.resized(lambda t: zoom_func(t))

    @staticmethod
    def zoom_out(clip, duration=None, zoom_from=1.2):
        """Apply zoom out effect"""
        if duration is None:
            duration = clip.duration / 2

        def zoom_func(t):
            progress = min(1, t / duration)
            zoom_factor = zoom_from - (zoom_from - 1) * progress
            return zoom_factor

        return clip.resized(lambda t: zoom_func(t))

    @staticmethod
    def cross_fade(clip1, clip2, duration=0.5):
        """Create a cross-fade transition between two clips"""
        # Ensure the clips can be crossfaded
        if duration > clip1.duration or duration > clip2.duration:
            duration = min(clip1.duration, clip2.duration) / 2

        # Apply fades
        clip1_out = VideoEffects.fade_out(clip1.subclipped(0, clip1.duration), duration)
        clip2_in = VideoEffects.fade_in(clip2, duration)

        # Create the crossfade
        crossfade = CompositeVideoClip([
            clip1_out,
            clip2_in.with_start(clip1.duration - duration)
        ])

        return crossfade

    @staticmethod
    def color_enhance(clip, saturation=1.2, contrast=1.1, brightness=1.0):
        """Apply color enhancement to the clip using PIL"""
        if not PIL_AVAILABLE or not NUMPY_AVAILABLE:
            logger.warning("PIL or NumPy not available. Cannot apply color enhancement.")
            return clip

        def enhance_frame(get_frame, t):
            frame = get_frame(t)

            # Convert to PIL Image
            img = Image.fromarray(frame)

            # Apply enhancements
            if brightness != 1.0:
                img = ImageEnhance.Brightness(img).enhance(brightness)
            if contrast != 1.0:
                img = ImageEnhance.Contrast(img).enhance(contrast)
            if saturation != 1.0:
                img = ImageEnhance.Color(img).enhance(saturation)

            # Convert back to numpy array
            return np.array(img)

        return clip.transform(enhance_frame)


class AudioEffects:
    """Modern audio effects compatible with MoviePy 2.1.2"""

    @staticmethod
    def audio_fade_in(clip, duration):
        """Apply audio fade in"""
        return clip.with_effects([AudioFadeIn(duration)])

    @staticmethod
    def audio_fade_out(clip, duration):
        """Apply audio fade out"""
        return clip.with_effects([AudioFadeOut(duration)])

    @staticmethod
    def normalize_audio(clip, target_dBFS=-14):
        """Normalize audio to target loudness"""
        if not PYDUB_AVAILABLE:
            logger.warning("Pydub not available. Using simple audio normalization.")
            return clip

        # Export to temporary file for processing
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_file.close()
        temp_path = temp_file.name

        try:
            # Write audio to file
            clip.write_audiofile(temp_path, logger=None)

            # Process with pydub
            audio = AudioSegment.from_file(temp_path)

            # Calculate required gain adjustment
            change = target_dBFS - audio.dBFS

            # Only apply normalization if needed
            if abs(change) > 0.5:
                normalized_audio = audio.apply_gain(change)

                # Export normalized audio
                norm_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                norm_file.close()
                norm_path = norm_file.name

                normalized_audio.export(norm_path, format="wav")

                # Create new audio clip
                new_clip = AudioFileClip(norm_path)

                # Clean up
                os.unlink(norm_path)

                return new_clip

            # No significant change needed
            return clip

        except Exception as e:
            logger.error(f"Error normalizing audio: {e}")
            return clip
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class AnimatedSubtitle:
    """Modern animated subtitle generator for short-form videos"""

    ANIMATION_STYLES = {
        "fade": "Simple fade in/out",
        "pop": "Pop animation with slight scaling",
        "typewriter": "Character-by-character reveal",
        "slide": "Slide in from bottom",
        "gradient": "Gradient text with sliding reveal"
    }

    def __init__(self, config: PipelineConfig):
        """Initialize subtitle generator with platform config"""
        self.config = config
        self.platform = config.platform

        # Set default style based on platform with larger fonts
        if self.platform in ['tiktok', 'instagram_reels']:
            self.default_animation = 'pop'
            self.fontsize = 56  # Increased from 42
        else:
            # More conservative for YouTube but still larger
            self.default_animation = 'fade'
            self.fontsize = 48  # Increased from 36

        # Check for local font in the fonts directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        font_path = os.path.join(project_root, "fonts", "Montserrat-VariableFont_wght.ttf")

        # Fallback fonts if local font not available
        self.font_options = ['Arial', 'Helvetica', 'DejaVuSans', 'Roboto']

        # Use local font if available, otherwise try system fonts
        if os.path.exists(font_path):
            self.font = font_path
            logger.info(f"Using local font: {font_path}")
        else:
            logger.warning(f"Local font not found at {font_path}, using system font")
            self.font = self.font_options[0]  # Use first system font as fallback

    def _get_platform_text_style(self):
        """Get subtitle styling based on platform"""
        # Base style for all platforms - modern approach with larger fonts
        style = {
            'font': self.font,
            'fontsize': self.fontsize,
            'color': 'white',  # Keep white for best contrast
            'method': 'caption',
            'align': 'center',
            'stroke_color': 'black',  # Add stroke for better contrast
            'stroke_width': 1.5,  # Moderate stroke width
        }

        # Platform specific adjustments with larger fonts
        if self.platform == 'tiktok':
            # TikTok - bold text with strong contrast
            style['fontsize'] = 56  # Increased from 42
            style['color'] = '#FFFFFF'  # Pure white
            style['stroke_width'] = 2.0  # Thicker stroke for TikTok
        elif self.platform == 'instagram_reels':
            # Instagram - clean, modern look
            style['fontsize'] = 52  # Increased from 40
            style['color'] = '#FFFFFF'
            style['stroke_width'] = 1.8  # Medium stroke for Instagram
        elif self.platform == 'youtube_shorts':
            # YouTube - slightly more conservative
            style['fontsize'] = 48  # Increased from 36
            style['stroke_width'] = 1.5  # Standard stroke for YouTube

        return style

    def create_subtitle_clip(self, text, duration, size, position='bottom', animation_style=None):
        """Create an animated subtitle clip with modern styling"""
        if not animation_style:
            animation_style = self.default_animation

        # Get styling for current platform
        style = self._get_platform_text_style()

        # Handle text size - subtract margins
        text_width = size[0] - 100  # 50px margins on each side

        # Create text clip with modern styling - NO background
        txt_clip = TextClip(
            text=text,
            font=style['font'],
            font_size=style['fontsize'],
            color=style['color'],
            method=style['method'],
            text_align=style['align'],
            size=(text_width, None),
            bg_color=None,  # No background!
            stroke_color=style.get('stroke_color', 'black'),
            stroke_width=int(style.get('stroke_width', 1.5))
        )

        # Set duration
        txt_clip = txt_clip.with_duration(duration)

        # Add a semi-transparent background behind text for better readability
        if self.config.platform in ['tiktok', 'instagram_reels']:
            # Create a slightly larger background for text with padding
            padding = 20  # Padding around text
            bg_width = txt_clip.size[0] + padding * 2
            bg_height = txt_clip.size[1] + padding * 2

            # Create background with black color
            bg_clip = ColorClip(
                size=(bg_width, bg_height),
                color=(0, 0, 0)  # Black color
            )

            # Add opacity using set_opacity
            bg_clip = bg_clip.with_opacity(0.4)  # 40% opacity
            bg_clip = bg_clip.with_duration(duration)

            # Composite text on background
            txt_clip = CompositeVideoClip([
                bg_clip,
                txt_clip.with_position('center')
            ])

        # Apply animations
        animated_clip = self._apply_animation(txt_clip, animation_style, duration)

        # Set position on the main video - position lower for better visibility
        if position == 'bottom':
            # Position subtitles closer to bottom for social media format
            final_y = size[1] - animated_clip.size[1] - 100  # 100px from bottom for better visibility
            final_position = ('center', final_y)
        elif position == 'top':
            final_position = ('center', 80)  # 80px from top
        elif position == 'center':
            final_position = 'center'
        else:
            final_position = position

        return animated_clip.with_position(final_position)

    def _apply_animation(self, clip, style, duration):
        """Apply animation to subtitle clip with modern effects"""
        # Animation durations - quick and snappy for social media
        anim_in_duration = min(0.25, duration * 0.15)
        anim_out_duration = min(0.2, duration * 0.1)

        # Apply different animations based on style
        if style == 'fade':
            # Simple fade in/out
            clip = VideoEffects.fade_in(clip, anim_in_duration)
            clip = VideoEffects.fade_out(clip, anim_out_duration)

        elif style == 'pop':
            # Scale animation with modern bouncy effect
            def scale_func(t):
                if t < anim_in_duration:
                    # Pop in with slight bounce effect
                    progress = t / anim_in_duration
                    if progress < 0.7:
                        return 0.5 + 0.5 * (progress / 0.7)
                    else:
                        return 1.0 + 0.08 * math.sin((progress - 0.7) * math.pi / 0.3)
                elif t > duration - anim_out_duration:
                    # Pop out - simple scale down
                    progress = (t - (duration - anim_out_duration)) / anim_out_duration
                    return 1.0 - 0.3 * progress
                else:
                    return 1.0

            # Apply scale animation
            clip = clip.resized(lambda t: scale_func(t))
            # Also add fade for smoothness
            clip = VideoEffects.fade_in(clip, anim_in_duration / 2)
            clip = VideoEffects.fade_out(clip, anim_out_duration)



        elif style == 'slide':
            # Slide from bottom with easing
            def slide_pos(t):
                if t < anim_in_duration:
                    # Slide in from bottom with ease out
                    progress = t / anim_in_duration
                    eased_progress = 1 - (1 - progress) ** 2.5  # Stronger ease out
                    offset = (1 - eased_progress) * 120  # Start 120px below
                    return ('center', offset)
                elif t > duration - anim_out_duration:
                    # Slide out to bottom with ease in
                    progress = (t - (duration - anim_out_duration)) / anim_out_duration
                    eased_progress = progress ** 2  # Ease in quad
                    offset = eased_progress * 100  # End 100px below
                    return ('center', offset)
                else:
                    return ('center', 0)  # Stay in place
            # Create a container clip with the right dimensions for the sliding effect
            container_h = clip.h + 120  # Make room for the slide
            container = ColorClip((clip.w, container_h), color=(0, 0, 0, 0))
            container = container.with_duration(duration)
            # Add the sliding subtitle to the container with explicit lambda
            # This is the key fix - ensuring we're properly wrapping the function
            position_clip = clip.with_position(lambda t: slide_pos(t))
            sliding_clip = CompositeVideoClip([
                container,
                position_clip
            ])
            # Also add fade for smoothness
            sliding_clip = VideoEffects.fade_in(sliding_clip, anim_in_duration / 3)
            sliding_clip = VideoEffects.fade_out(sliding_clip, anim_out_duration / 2)
            return sliding_clip

        elif style == 'typewriter':
            # Modern typewriter effect
            def typewriter_crop(t):
                if t < anim_in_duration:
                    progress = t / anim_in_duration
                    # Ease out for more natural typing feel
                    eased_progress = 1 - (1 - progress) ** 2
                    width = int(clip.w * eased_progress)
                    x1 = 0
                    y1 = 0
                    x2 = width
                    y2 = clip.h
                    return (x1, y1, x2, y2)
                return (0, 0, clip.w, clip.h)

            # Create typewriter effect
            clip = clip.crop(typewriter_crop)

            # Add fade out
            clip = VideoEffects.fade_out(clip, anim_out_duration)

        elif style == 'gradient':
            # New modern gradient style with reveal
            def gradient_mask(t):
                if t < anim_in_duration:
                    progress = t / anim_in_duration
                    eased_progress = 1 - (1 - progress) ** 2

                    # Create a gradient mask effect
                    width = clip.w
                    height = clip.h
                    mask = np.zeros((height, width), dtype=np.uint8)

                    # Calculate gradient width
                    reveal_width = int(width * eased_progress)
                    gradient_width = min(int(width * 0.2), 30)  # 20% width or 30px max

                    # Fill the revealed part
                    if reveal_width > 0:
                        mask[:, :reveal_width] = 255

                    # Create gradient at the edge
                    if reveal_width < width:
                        gradient_end = min(reveal_width + gradient_width, width)
                        gradient_range = gradient_end - reveal_width
                        if gradient_range > 0:
                            for x in range(reveal_width, gradient_end):
                                alpha = 255 * (1 - (x - reveal_width) / gradient_range)
                                mask[:, x] = alpha

                    return mask
                else:
                    # Full visibility after animation
                    return np.ones((clip.h, clip.w), dtype=np.uint8) * 255

            # Add gradient reveal effect with image clip
            # Simplified implementation without actual image processing since it requires numpy
            # Just use fade for simplicity when numpy isn't available
            if NUMPY_AVAILABLE:
                clip = clip.with_mask(lambda t: gradient_mask(t))
            else:
                clip = VideoEffects.fade_in(clip, anim_in_duration)

            # Add fade out
            clip = VideoEffects.fade_out(clip, anim_out_duration)

        return clip

    def chunk_subtitles(self, text, max_chars=50):
        """Split long text into smaller chunks for better readability

        Args:
            text: The full subtitle text
            max_chars: Maximum characters per chunk

        Returns:
            List of text chunks
        """
        # If text is short enough, return as is
        if len(text) <= max_chars:
            return [text]

        # For short-form videos, we want to break at logical points
        # like sentence endings or commas where possible
        chunks = []

        # First try to split by sentences
        sentences = []
        for sentence in text.replace('! ', '! SPLIT').replace('? ', '? SPLIT').replace('. ', '. SPLIT').split('SPLIT'):
            sentences.append(sentence.strip())

        current_chunk = []
        current_length = 0

        # Process each sentence
        for sentence in sentences:
            # If sentence is short enough, add it to the current chunk
            if current_length + len(sentence) + 1 <= max_chars:
                current_chunk.append(sentence)
                current_length += len(sentence) + 1  # +1 for space
            elif len(sentence) > max_chars:
                # If the sentence itself is too long, we need to split it by clauses
                # First, add any accumulated chunks
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Try to split by clauses (at commas, semicolons, etc.)
                clauses = []
                for clause in sentence.replace(', ', ', SPLIT').replace('; ', '; SPLIT').split('SPLIT'):
                    clauses.append(clause.strip())

                for clause in clauses:
                    if len(clause) <= max_chars:
                        chunks.append(clause)
                    else:
                        # If clause is still too long, fall back to word splitting
                        words = clause.split()
                        clause_chunk = []
                        clause_length = 0

                        for word in words:
                            if clause_length + len(word) + 1 <= max_chars:
                                clause_chunk.append(word)
                                clause_length += len(word) + 1
                            else:
                                if clause_chunk:
                                    chunks.append(' '.join(clause_chunk))
                                clause_chunk = [word]
                                clause_length = len(word)

                        if clause_chunk:
                            chunks.append(' '.join(clause_chunk))
            else:
                # This sentence would make the chunk too long, save current chunk and start a new one
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)

        # Add the last chunk if any
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks


class VideoAssembler:
    """Enhanced video assembler with modern effects and subtitles"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = os.path.join(OUTPUT_DIR, "videos")
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize subtitle generator
        self.subtitle_generator = AnimatedSubtitle(config)

        # Set platform specific settings
        self.platform = config.platform
        self.platform_config = config.platform_config
        self.visual_config = config.visual

        # Visual settings
        if self.visual_config.color_scheme == 'vibrant':
            self.color_saturation = 1.2
            self.color_contrast = 1.1
        elif self.visual_config.color_scheme == 'muted':
            self.color_saturation = 0.9
            self.color_contrast = 1.0
        else:  # professional or default
            self.color_saturation = 1.0
            self.color_contrast = 1.05

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

    def _apply_visual_effects(self, clip, scene_index, total_scenes):
        """Apply platform-optimized visual effects to the clip

        Args:
            clip: Video clip to enhance
            scene_index: Current scene index (for varied effects)
            total_scenes: Total number of scenes

        Returns:
            Enhanced clip
        """
        # Skip effects for very short clips
        if not clip.duration:
            return clip
        if clip.duration < 1.0:
            return clip

        # Apply platform and mood specific effects
        if self.platform in ['tiktok', 'instagram_reels']:
            # More dynamic effects for these platforms
            if self.visual_config.motion_effects:
                # Apply different effects based on scene position
                if scene_index == 0:
                    # First scene - zoom in or pan effect
                    clip = VideoEffects.zoom_in(clip, clip.duration / 2, zoom_to=1.1)
                elif scene_index == total_scenes - 1:
                    # Last scene - subtle zoom out
                    clip = VideoEffects.zoom_out(clip, clip.duration / 2, zoom_from=1.1)
                else:
                    # Middle scenes - alternate between effects
                    if scene_index % 3 == 0:
                        # Zoom in
                        clip = VideoEffects.zoom_in(clip, clip.duration, zoom_to=1.08)
                    elif scene_index % 3 == 1:
                        # Zoom out
                        clip = VideoEffects.zoom_out(clip, clip.duration, zoom_from=1.08)

            # Apply color enhancements
            if self.visual_config.apply_filters:
                clip = VideoEffects.color_enhance(
                    clip,
                    saturation=self.color_saturation,
                    contrast=self.color_contrast
                )

        else:  # youtube_shorts or general
            # More subtle effects
            if self.visual_config.motion_effects:
                # Apply subtle zoom for longer scenes
                if clip.duration > 3.0:
                    zoom_amount = 1.05  # Very subtle
                    clip = VideoEffects.zoom_in(clip, clip.duration, zoom_to=zoom_amount)

            # Apply more moderate color enhancements
            if self.visual_config.apply_filters:
                clip = VideoEffects.color_enhance(
                    clip,
                    saturation=self.color_saturation * 0.9,  # More subtle
                    contrast=self.color_contrast
                )

        return clip

    def _apply_transition(self, clip1, clip2, transition_type, duration=0.5):
        """Apply transition effect between clips with improved timing safety

        Args:
            clip1: First clip
            clip2: Second clip
            transition_type: Type of transition
            duration: Transition duration

        Returns:
            Combined clip with transition
        """
        # Default to crossfade for very short clips with safer duration
        if clip1.duration < duration * 2 or clip2.duration < duration * 2:
            duration = min(min(clip1.duration, clip2.duration) / 5, 0.3)  # Max 300ms, 1/5 of shortest clip
            transition_type = "fade"

        # Define a tiny gap between clips (in seconds)
        gap_duration = 0.05  # 50ms gap

        # Apply transition
        if transition_type == "fade" or transition_type == "crossfade":
            # Apply fade out to clip1
            clip1_faded = VideoEffects.fade_out(clip1, duration)

            # Apply fade in to clip2
            clip2_faded = VideoEffects.fade_in(clip2, duration)

            # Position clips with a gap instead of overlap
            clip2_start = clip1.duration + gap_duration

            # Create composite with explicit timing to ensure gap
            transition = CompositeVideoClip([
                clip1_faded,
                clip2_faded.with_start(clip2_start)
            ], size=clip1.size)

            # Handle audio separately with gap
            if hasattr(clip1, 'audio') and clip1.audio and hasattr(clip2, 'audio') and clip2.audio:
                try:
                    # Apply audio fades
                    audio1 = clip1.audio
                    audio2 = clip2.audio

                    # Only apply fade out if audio1 is long enough
                    if audio1.duration > duration * 2:
                        audio1 = AudioEffects.audio_fade_out(audio1, duration)

                    # Apply fade in to audio2
                    audio2 = AudioEffects.audio_fade_in(audio2, duration)

                    # Create composite audio with explicit timing to ensure gap
                    combined_audio = CompositeAudioClip([
                        audio1,
                        audio2.with_start(clip2_start)
                    ])

                    # Set audio on transition
                    transition = transition.with_audio(combined_audio)
                except Exception as e:
                    logger.warning(f"Audio transition failed, falling back to simple audio: {e}")
                    # If audio transition fails, keep the video transition but use simpler audio
                    if clip2.audio:
                        transition = transition.with_audio(clip2.audio)

            return transition

        elif transition_type == "cut":
            # Simple cut with gap - use CompositeVideoClip with explicit timing
            clip2_start = clip1.duration + gap_duration
            return CompositeVideoClip([
                clip1,
                clip2.with_start(clip2_start)
            ], size=clip1.size)

        else:
            # Default to crossfade
            logger.warning(f"Unknown transition type: {transition_type}. Using crossfade.")
            return self._apply_transition(clip1, clip2, "fade", duration)

    def _process_audio(self, audio_clip):
        """Enhance audio clip for better quality with safety checks

        Args:
            audio_clip: Audio clip to enhance

        Returns:
            Enhanced audio clip
        """
        if audio_clip is None:
            return None

        # Apply audio normalization - critical for mobile viewing
        # enhanced_audio = AudioEffects.normalize_audio(audio_clip)
        enhanced_audio = audio_clip

        # Calculate safe fade durations based on clip length
        # Use very minimal fades to avoid timing issues
        clip_duration = enhanced_audio.duration
        fade_in_duration = min(0.03, clip_duration * 0.03)  # 3% of duration or 30ms, whichever is smaller
        fade_out_duration = min(0.05, clip_duration * 0.05)  # 5% of duration or 50ms, whichever is smaller

        # Apply fades only if duration is sufficient
        if clip_duration > fade_in_duration * 2:
            enhanced_audio = AudioEffects.audio_fade_in(enhanced_audio, fade_in_duration)

        if clip_duration > fade_out_duration * 2:
            enhanced_audio = AudioEffects.audio_fade_out(enhanced_audio, fade_out_duration)

        return enhanced_audio

    def _add_subtitles(self, video_clip, text, duration, animation_style=None):
        """Add animated subtitles to video clip

        Args:
            video_clip: Video clip to add subtitles to
            text: Subtitle text
            duration: Duration for subtitles
            animation_style: Style of animation for subtitles

        Returns:
            Video with subtitles
        """
        # Get video dimensions
        width, height = video_clip.size

        # For short form content, chunk text for better readability
        # Use shorter text chunks for mobile-first platforms
        if self.platform == 'tiktok':
            max_chars = 35  # Very short for TikTok's fast pace
        elif self.platform == 'instagram_reels':
            max_chars = 38  # Slightly longer for Instagram
        elif self.platform == 'youtube_shorts':
            max_chars = 42  # Longer for YouTube
        else:
            max_chars = 50  # Default for other platforms

        text_chunks = self.subtitle_generator.chunk_subtitles(text, max_chars)

        if not text_chunks:
            return video_clip

        # If single chunk, create one subtitle
        if len(text_chunks) == 1:
            subtitle = self.subtitle_generator.create_subtitle_clip(
                text_chunks[0],
                duration,
                (width, height),
                position='bottom',
                animation_style=animation_style
            )

            # Add subtitle to video
            return CompositeVideoClip([video_clip, subtitle])

        # For multiple chunks, calculate timing
        chunks_duration = duration / len(text_chunks)
        subtitle_clips = []

        # Generate subtitle for each chunk
        for i, chunk in enumerate(text_chunks):
            # Calculate timing for this chunk
            chunk_start = i * chunks_duration
            chunk_duration = min(chunks_duration, duration - chunk_start)

            # Create subtitle clip
            subtitle = self.subtitle_generator.create_subtitle_clip(
                chunk,
                chunk_duration,
                (width, height),
                position='bottom',
                animation_style=animation_style
            )

            # Position in time
            subtitle = subtitle.with_start(chunk_start)

            subtitle_clips.append(subtitle)

        # Add all subtitles to video
        return CompositeVideoClip([video_clip] + subtitle_clips)

    def assemble_video(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble final video from processed scenes with improved timing synchronization"""
        processed_scenes = state.get("processed_scenes", [])
        script = state.get("script", {})
        hook_audio_path = state.get("hook_audio_path")

        # Create a unique output filename
        output_filename = f"video_{int(time.time())}.mp4"
        output_path = os.path.join(self.output_dir, output_filename)

        try:
            # Filter out scenes with missing assets
            valid_scenes = [
                scene for scene in processed_scenes
                if "image_path" in scene and os.path.exists(scene["image_path"]) and
                   "voice_path" in scene and os.path.exists(scene["voice_path"])
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

            # Create hook clip if hook audio exists
            hook_clip = None
            if hook_audio_path and os.path.exists(hook_audio_path) and "hook" in script:
                try:
                    # Load hook audio
                    hook_audio = AudioFileClip(hook_audio_path)

                    # Precise hook duration based on audio
                    hook_duration = hook_audio.duration

                    # Use dedicated hook image if available, otherwise fall back to first scene's image
                    hook_image_path = state.get("hook_image_path")
                    if not hook_image_path or not os.path.exists(hook_image_path):
                        hook_image_path = valid_scenes[0]["image_path"] if valid_scenes else None
                        logger.info("Using first scene image for hook (dedicated hook image not found)")
                    else:
                        logger.info(f"Using dedicated hook image: {hook_image_path}")

                    if hook_image_path and os.path.exists(hook_image_path):
                        # Create image clip for hook
                        hook_img = ImageClip(hook_image_path)

                        # Resize and center as needed
                        if hook_img.size[0] / hook_img.size[1] > target_width / target_height:
                            hook_img = hook_img.resized(height=target_height)
                        else:
                            hook_img = hook_img.resized(width=target_width)

                        # Add background
                        bg_clip = ColorClip(size=(target_width, target_height), color=(0, 0, 0))
                        bg_clip = bg_clip.with_duration(hook_duration)

                        # Composite
                        hook_img = CompositeVideoClip([
                            bg_clip,
                            hook_img.with_position('center')
                        ])

                        # Apply visual effects for hook to be attention-grabbing
                        hook_img = self._apply_visual_effects(hook_img, -1, len(valid_scenes) + 1)

                        # Set hook audio
                        hook_clip = hook_img.with_duration(hook_duration).with_audio(hook_audio)

                        # Add hook subtitles
                        hook_text = script["hook"]
                        hook_clip = self._add_subtitles(
                            hook_clip,
                            hook_text,
                            hook_duration,
                            animation_style='pop'  # Attention-grabbing for hook
                        )

                        # Add to video clips at the beginning
                        video_clips.append(hook_clip)

                        logger.info(f"Added spoken hook with duration: {hook_duration:.2f}s")
                except Exception as e:
                    logger.error(f"Error creating hook clip: {e}")
                    logger.error(traceback.format_exc())

            # Process each scene
            for i, scene in enumerate(valid_scenes):
                try:
                    # First load and analyze the audio to get exact duration
                    # This is critical to ensure proper timing synchronization
                    audio_clip = AudioFileClip(scene["voice_path"])

                    # Add a small safety margin to audio duration (trim 20ms from end)
                    precise_duration = max(audio_clip.duration - 0.02, 0.1)  # Ensure minimum 0.1s

                    # Apply precise duration to the scene for consistency
                    scene["precise_duration"] = precise_duration

                    # For audio processing, we apply a slightly shorter duration
                    # to prevent the timing error
                    audio_clip = audio_clip.subclipped(0, precise_duration)

                    # Process audio for better quality
                    enhanced_audio = self._process_audio(audio_clip)

                    # Now load image and set it to the exact audio duration
                    img_clip = ImageClip(scene["image_path"])

                    # Resize to target dimensions while preserving aspect ratio
                    # First get image dimensions
                    img_width, img_height = img_clip.size

                    # Determine scaling approach based on aspect ratio
                    if img_width / img_height > target_width / target_height:
                        # Image is wider than target - scale by height
                        img_clip = img_clip.resized(height=target_height)
                    else:
                        # Image is taller than target - scale by width
                        img_clip = img_clip.resized(width=target_width)

                    # Get new dimensions after resizing
                    new_width, new_height = img_clip.size

                    # Create background of exact target dimensions
                    bg_clip = ColorClip(size=(target_width, target_height), color=(0, 0, 0))
                    bg_clip = bg_clip.with_duration(precise_duration)

                    # Composite image on background
                    img_clip = CompositeVideoClip([
                        bg_clip,
                        img_clip.with_position('center')
                    ])

                    # Set duration to match audio exactly
                    img_clip = img_clip.with_duration(precise_duration)

                    # Apply visual effects based on platform and position
                    img_clip = self._apply_visual_effects(img_clip, i, len(valid_scenes))

                    # Set enhanced audio
                    video_clip = img_clip.with_audio(enhanced_audio)

                    # Always add subtitles for the spoken content
                    if "text" in scene and scene["text"]:
                        # Determine animation style based on scene position
                        if i == 0:
                            # First scene - more attention-grabbing
                            animation_style = 'pop'
                        elif i == len(valid_scenes) - 1:
                            # Last scene - fade for conclusion
                            animation_style = 'fade'
                        else:
                            # Middle scenes - vary styles
                            animation_styles = ['slide', 'fade', 'pop', 'gradient', 'typewriter']
                            animation_style = animation_styles[i % len(animation_styles)]

                        # Add animated subtitles - use precise_duration for consistent timing
                        video_clip = self._add_subtitles(
                            video_clip,
                            scene["text"],
                            precise_duration,
                            animation_style
                        )

                    # Add clip to list
                    video_clips.append(video_clip)

                except Exception as e:
                    logger.error(f"Error processing scene {i}: {e}")
                    logger.error(traceback.format_exc())

            if not video_clips:
                logger.error("Failed to create any valid video clips")
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
                # Create transitions between clips
                transitions = [scene.get("transition", "fade") for scene in valid_scenes[:-1]]

                # Apply transitions sequentially with safer timing
                result_clip = video_clips[0]

                for i in range(len(video_clips) - 1):
                    try:
                        # Get transition type
                        transition_type = transitions[i] if i < len(transitions) else "fade"

                        # Apply transition to next clip with a slightly shorter overlap
                        # to prevent timing issues
                        transition_duration = min(0.4, min(result_clip.duration, video_clips[i + 1].duration) / 4)

                        # Apply transition with safe duration
                        result_clip = self._apply_transition(
                            result_clip,
                            video_clips[i + 1],
                            transition_type,
                            duration=transition_duration
                        )
                    except Exception as e:
                        logger.error(f"Error applying transition {i}: {e}")
                        logger.error(traceback.format_exc())
                        # Try simple concatenation as fallback
                        try:
                            result_clip = concatenate_videoclips([result_clip, video_clips[i + 1]])
                        except:
                            logger.error(f"Failed to concatenate clips {i} and {i + 1}")
                            break

                final_clips = [result_clip]

            # Concatenate if multiple clips
            if len(final_clips) > 1:
                try:
                    final_video = concatenate_videoclips(final_clips)
                except Exception as e:
                    logger.error(f"Error concatenating final clips: {e}")
                    final_video = final_clips[0]
            else:
                final_video = final_clips[0]

            # Write final video using a try-except with several fallback strategies
            logger.info(f"Writing video to {output_path}...")

            # Create temporary directory for intermediate files
            temp_dir = tempfile.mkdtemp()
            temp_audio = os.path.join(temp_dir, 'temp_audio.m4a')

            try:
                # First attempt - use AV1 with reasonable quality
                final_video.write_videofile(
                    output_path,
                    codec='libx264',
                    # codec='libaom-av1',  # AV1 codec
                    audio_codec='aac',
                    temp_audiofile=temp_audio,
                    remove_temp=True,
                    threads=4,
                    fps=self.config.platform_config.framerate,
                    preset='medium',
                    bitrate='3000k',  # Reasonable bitrate for AV1 (lower than H.264 for similar quality)
                    ffmpeg_params=['-crf', '30']  # Constant Rate Factor - lower is better quality (range 0-63 for AV1)
                )
            except Exception as e:
                # Check if the error is related to audio timing
                if "Accessing time t=" in str(e) and "with clip duration=" in str(e):
                    logger.warning("Audio timing error detected. Trying with audio preprocessing...")

                    try:
                        # Try extracting and preprocessing audio separately
                        temp_audio_path = os.path.join(temp_dir, 'preprocessed_audio.wav')

                        # Write audio with safety margins
                        if hasattr(final_video, 'audio') and final_video.audio:
                            # Get audio and apply a safety margin
                            safe_audio = final_video.audio.subclipped(0, final_video.audio.duration - 0.05)
                            safe_audio.write_audiofile(temp_audio_path, fps=44100)

                            # Create a new video without audio
                            video_without_audio = final_video.without_audio()

                            # Create a new audio clip from the safely processed file
                            processed_audio = AudioFileClip(temp_audio_path)

                            # Combine video with processed audio
                            final_video = video_without_audio.with_audio(processed_audio)

                        # Try writing with the preprocessed audio, using AV1
                        final_video.write_videofile(
                            output_path,
                            codec='libx264',
                            # codec='libaom-av1',
                            audio_codec='aac',
                            temp_audiofile=temp_audio,
                            remove_temp=True,
                            threads=2,
                            fps=24,
                            preset='medium',
                            bitrate='2000k',  # Lower bitrate for faster encoding in fallback
                            ffmpeg_params=['-crf', '35']  # Slightly lower quality for speed in fallback
                        )
                    except Exception as e2:
                        logger.error(f"Second attempt also failed: {e2}")

                        # Final fallback - try without audio processing
                        try:
                            logger.info("Trying with simpler parameters and without audio processing...")
                            final_video.write_videofile(
                                output_path,
                                codec='libx264',
                                audio_codec='aac',
                                threads=1,
                                fps=24,
                                preset='ultrafast',
                                write_logfile=True
                            )
                        except Exception as e3:
                            logger.error(f"All video export attempts failed: {e3}")
                            return {
                                "success": False,
                                "error": f"Failed to write video file after multiple attempts: {str(e3)}",
                                "state": state
                            }
                else:
                    # For other errors, try with simpler parameters
                    logger.info("Retrying with simpler video export settings...")
                    try:
                        # Fallback to SVT-AV1 with faster preset
                        final_video.write_videofile(
                            output_path,
                            codec='libsvtav1',
                            audio_codec='aac',
                            threads=2,
                            fps=24,
                            preset='10',  # SVT-AV1 preset (0-13, higher is faster)
                            bitrate='2000k',
                            ffmpeg_params=['-crf', '38']  # Compromise between quality and speed
                        )
                    except Exception as e2:
                        logger.error(f"Second attempt at writing video also failed: {e2}")
                        return {
                            "success": False,
                            "error": f"Failed to write video file: {str(e2)}",
                            "state": state
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
                "state": state
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
            "script": state.get("script", {}),
            "content_strategy": state.get("content_strategy", {})
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
                "description": state.get("script", {}).get("hook", ""),
                "hashtags": [f"#{self.config.platform}"],
                "category": "Entertainment",
                "platform": self.config.platform,
                "video_path": assembly_result.get("video_path", ""),
                "duration": assembly_result.get("duration", 0)
            }

            return basic_metadata

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble video and generate metadata."""
        # Assemble video
        logger.info("Assembling final video with enhanced effects and subtitles...")
        assembly_result = self.assemble_video(state)

        if not assembly_result.get("success", False):
            logger.error(f"Video assembly failed: {assembly_result.get('error', 'Unknown error')}")
            result = state.copy()
            result["success"] = False
            result["error"] = assembly_result.get("error", "Unknown error")
            return result

        # Generate metadata
        logger.info("Generating platform-optimized video metadata...")
        metadata = self.generate_metadata(state, assembly_result)

        # Update state with video path and metadata
        result = state.copy()
        result["success"] = True
        result["video_path"] = assembly_result["video_path"]
        result["metadata"] = metadata

        return result
