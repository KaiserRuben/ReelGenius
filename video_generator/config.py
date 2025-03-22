
import os
from typing import Optional, Dict, List, Union, Any, Literal
from pydantic import BaseModel, Field, validator, model_validator
from pathlib import Path
import json
from loguru import logger
import dotenv

# Configure logging
logger.add("logs/video_generator_{time}.log", rotation="500 MB", level="INFO")

# Load environment variables
dotenv.load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
HISTORY_DIR = DATA_DIR / "history"
TEMPLATE_DIR = BASE_DIR / "video_generator" / "templates"

# Ensure directories exist
for directory in [
    DATA_DIR, OUTPUT_DIR, DATA_DIR / "inputs", HISTORY_DIR, 
    OUTPUT_DIR / "videos", OUTPUT_DIR / "metadata", OUTPUT_DIR / "temp"
]:
    directory.mkdir(exist_ok=True, parents=True)

# Platform types
PlatformType = Literal["tiktok", "youtube_shorts", "instagram_reels", "general"]

class PlatformConfig(BaseModel):
    """Platform-specific configuration."""
    name: PlatformType = Field(..., description="Platform name")
    aspect_ratio: str = Field("9:16", description="Video aspect ratio")
    min_duration: int = Field(15, description="Minimum video duration in seconds")
    max_duration: int = Field(60, description="Maximum video duration in seconds")
    resolution: str = Field("1080x1920", description="Video resolution")
    framerate: int = Field(30, description="Video frame rate")
    audio_quality: str = Field("high", description="Audio quality (low, medium, high)")
    captions: bool = Field(True, description="Whether to include captions")
    text_overlay: bool = Field(True, description="Whether to include text overlays")
    watermark: bool = Field(False, description="Whether to include watermark")
    optimize_hook: bool = Field(True, description="Whether to optimize for strong hook")
    hook_duration: int = Field(3, description="Duration of hook in seconds")
    audience_attention_span: int = Field(5, description="Typical audience attention span in seconds")
    
    @validator('resolution')
    def validate_resolution(cls, v):
        """Validate resolution format."""
        try:
            width, height = map(int, v.split('x'))
            return f"{width}x{height}"
        except:
            raise ValueError("Resolution must be in format WIDTHxHEIGHT (e.g., 1080x1920)")
    
    @model_validator(mode='after')
    def validate_duration(self):
        """Validate min and max duration."""
        if self.min_duration > self.max_duration:
            raise ValueError("min_duration cannot be greater than max_duration")
        return self

# Default platform configurations
PLATFORM_CONFIGS = {
    "tiktok": PlatformConfig(
        name="tiktok",
        aspect_ratio="9:16",
        min_duration=15,
        max_duration=60,
        resolution="1080x1920",
        framerate=30,
        optimize_hook=True,
        hook_duration=3,
        audience_attention_span=5
    ),
    "youtube_shorts": PlatformConfig(
        name="youtube_shorts",
        aspect_ratio="9:16",
        min_duration=15,
        max_duration=60,
        resolution="1080x1920",
        framerate=30,
        optimize_hook=True,
        hook_duration=5,
        audience_attention_span=8
    ),
    "instagram_reels": PlatformConfig(
        name="instagram_reels",
        aspect_ratio="9:16",
        min_duration=15,
        max_duration=90,
        resolution="1080x1920",
        framerate=30,
        optimize_hook=True,
        hook_duration=3,
        audience_attention_span=4
    ),
    "general": PlatformConfig(
        name="general",
        aspect_ratio="16:9",
        min_duration=30,
        max_duration=300,
        resolution="1920x1080",
        framerate=30,
        optimize_hook=False,
        hook_duration=0,
        audience_attention_span=15
    )
}

class LLMConfig(BaseModel):
    """LLM configuration."""
    api_key: str = Field(default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY", ""))
    model: str = Field("deepseek-chat", description="DeepSeek model name")
    base_url: str = Field("https://api.deepseek.com", description="DeepSeek API base URL")
    temperature: float = Field(0.7, description="Temperature for generation")
    max_tokens: int = Field(2000, description="Maximum tokens to generate")
    timeout: int = Field(120, description="API timeout in seconds")
    few_shot_examples: bool = Field(True, description="Whether to use few-shot examples")
    chain_of_thought: bool = Field(True, description="Whether to encourage chain of thought reasoning")
    use_meta_prompting: bool = Field(True, description="Whether to use meta-prompting")

class TTSConfig(BaseModel):
    """Text-to-speech configuration."""
    provider: str = Field("elevenlabs", description="TTS provider (elevenlabs, openai)")
    api_key: str = Field(default_factory=lambda: os.environ.get("ELEVENLABS_API_KEY", ""))
    voice_id: str = Field("21m00Tcm4TlvDq8ikWAM", description="Voice ID to use")
    speaking_rate: float = Field(1.1, description="Speaking rate multiplier for pacing")
    optimize_for_platform: bool = Field(True, description="Whether to optimize voice for platform")
    voice_style: str = Field("natural", description="Voice style (natural, enthusiastic, serious)")
    model: str = Field("eleven_turbo_v2", description="TTS model to use")

class ImageGenConfig(BaseModel):
    """Image generation configuration."""
    provider: str = Field("stability", description="Image provider (stability, openai)")
    api_key: str = Field(default_factory=lambda: os.environ.get("STABILITY_API_KEY", ""))
    model: str = Field("sd3", description="Model to use (sd3, sdxl)")
    style: str = Field("photorealistic", description="Image style")
    format: str = Field("portrait", description="Image format")
    quality: str = Field("standard", description="Image quality")
    candidates_per_prompt: int = Field(3, description="Number of image candidates to generate per prompt")
    use_image_evaluation: bool = Field(True, description="Whether to evaluate and select best images")

class VisualConfig(BaseModel):
    """Visual styling configuration."""
    color_scheme: str = Field("vibrant", description="Color scheme (vibrant, muted, professional, etc.)")
    text_animation: bool = Field(True, description="Whether to animate text")
    motion_effects: bool = Field(True, description="Whether to use motion effects like zooms")
    transition_style: str = Field("smooth", description="Transition style (smooth, sharp, creative)")
    visual_consistency: bool = Field(True, description="Whether to enforce visual consistency")
    apply_filters: bool = Field(True, description="Whether to apply visual filters")
    filter_intensity: float = Field(0.3, description="Intensity of filters (0.0-1.0)")

class PipelineConfig(BaseModel):
    """Main pipeline configuration."""
    platform: PlatformType = Field("tiktok", description="Target platform")
    platform_config: PlatformConfig = Field(default_factory=lambda: PLATFORM_CONFIGS["tiktok"])
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    image_gen: ImageGenConfig = Field(default_factory=ImageGenConfig)
    visual: VisualConfig = Field(default_factory=VisualConfig)
    history_tracking: bool = Field(True, description="Whether to track content history")
    quality_threshold: float = Field(0.7, description="Minimum quality score (0-1) for videos")
    max_retries: int = Field(3, description="Maximum retries for failed steps")
    parallel_processing: bool = Field(True, description="Whether to enable parallel processing")
    enable_detailed_logging: bool = Field(True, description="Whether to enable detailed logging")
    
    @model_validator(mode='after')
    def set_platform_config(self):
        """Set platform config based on platform."""
        if self.platform and self.platform in PLATFORM_CONFIGS:
            self.platform_config = PLATFORM_CONFIGS[self.platform]
        return self

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "PipelineConfig":
        """Load configuration from a JSON file."""
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"Config file {file_path} not found. Using defaults.")
            return cls()
        
        try:
            with open(file_path, "r") as f:
                config_data = json.load(f)
            return cls(**config_data)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return cls()
    
    def to_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to a JSON file."""
        file_path = Path(file_path)
        with open(file_path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)
        
    def validate_api_keys(self) -> bool:
        """Validate that all required API keys are present."""
        missing_keys = []
        
        if not self.llm.api_key:
            missing_keys.append("DEEPSEEK_API_KEY")
        
        if not self.tts.api_key and self.tts.provider == "elevenlabs":
            missing_keys.append("ELEVENLABS_API_KEY")
        
        if not self.image_gen.api_key and self.image_gen.provider == "stability":
            missing_keys.append("STABILITY_API_KEY")
        
        if missing_keys:
            logger.error(f"Missing API keys: {', '.join(missing_keys)}")
            return False
        
        return True
        
    def get_prompts_data(self) -> Dict[str, Any]:
        """Get data for prompt templates based on configuration."""
        return {
            "platform": self.platform,
            "platform_config": self.platform_config.model_dump(),
            "llm": self.llm.model_dump(),
            "visual": self.visual.model_dump(),
            "image_gen": self.image_gen.model_dump(),
            "tts": self.tts.model_dump()
        }

# Default configuration
DEFAULT_CONFIG = PipelineConfig()


