import os
from typing import Optional, Dict, List, Union, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from pathlib import Path
import json
from loguru import logger
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# Configure logging with rotation
os.makedirs("logs", exist_ok=True)
logger.add(
    "logs/video_generator_{time}.log",
    rotation="500 MB",
    retention="7 days",
    level=os.environ.get("LOG_LEVEL", "INFO"),
    compression="zip"
)


# Base directories with path validation
def get_validated_dir(path: Path) -> Path:
    """Ensure directory exists and is writable."""
    try:
        path.mkdir(exist_ok=True, parents=True)

        # Check if directory is writable
        test_file = path / ".write_test"
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            test_file.unlink()
        except (IOError, PermissionError) as e:
            logger.warning(f"Directory {path} is not writable: {e}")
    except Exception as e:
        logger.error(f"Error creating directory {path}: {e}")

    return path


BASE_DIR = Path(__file__).parent.parent
DATA_DIR = get_validated_dir(BASE_DIR / "data")
OUTPUT_DIR = get_validated_dir(BASE_DIR / "output")
HISTORY_DIR = get_validated_dir(DATA_DIR / "history")
TEMPLATE_DIR = get_validated_dir(BASE_DIR / "video_generator" / "templates")

# Ensure output subdirectories exist
for directory in [
    get_validated_dir(DATA_DIR / "inputs"),
    get_validated_dir(OUTPUT_DIR / "videos"),
    get_validated_dir(OUTPUT_DIR / "metadata"),
    get_validated_dir(OUTPUT_DIR / "temp"),
    get_validated_dir(OUTPUT_DIR / "media_assets")
]:
    pass  # Directory creation happens in get_validated_dir

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

    @field_validator('resolution')
    @classmethod
    def validate_resolution(cls, v):
        """Validate resolution format."""
        try:
            width, height = map(int, v.split('x'))
            return f"{width}x{height}"
        except:
            logger.warning(f"Invalid resolution format: {v}. Using default 1080x1920.")
            return "1080x1920"

    @field_validator('aspect_ratio')
    @classmethod
    def validate_aspect_ratio(cls, v):
        """Validate aspect ratio format."""
        try:
            width, height = map(int, v.split(':'))
            if width <= 0 or height <= 0:
                raise ValueError("Aspect ratio components must be positive numbers")
            return f"{width}:{height}"
        except:
            logger.warning(f"Invalid aspect ratio format: {v}. Using default 9:16.")
            return "9:16"

    @model_validator(mode='after')
    def validate_duration(self):
        """Validate min and max duration."""
        if self.min_duration > self.max_duration:
            logger.warning(
                f"min_duration ({self.min_duration}) is greater than max_duration ({self.max_duration}). Swapping values.")
            self.min_duration, self.max_duration = self.max_duration, self.min_duration

        # Enforce reasonable limits
        if self.min_duration < 5:
            logger.warning(f"min_duration ({self.min_duration}) is too small. Setting to 5 seconds.")
            self.min_duration = 5

        if self.max_duration > 600:
            logger.warning(f"max_duration ({self.max_duration}) is too large. Setting to 600 seconds (10 minutes).")
            self.max_duration = 600

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
    max_tokens: int = Field(5000, description="Maximum tokens to generate")
    timeout: int = Field(int(os.environ.get("LLM_TIMEOUT", "120")), description="API timeout in seconds")
    few_shot_examples: bool = Field(True, description="Whether to use few-shot examples")
    chain_of_thought: bool = Field(True, description="Whether to encourage chain of thought reasoning")
    use_meta_prompting: bool = Field(False, description="Whether to use meta-prompting")

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        """Validate temperature is in valid range."""
        if v < 0.0 or v > 1.0:
            logger.warning(f"Temperature {v} is outside valid range (0.0-1.0). Clamping to valid range.")
            return max(0.0, min(v, 1.0))
        return v

    @field_validator('timeout')
    @classmethod
    def validate_timeout(cls, v):
        """Validate timeout is reasonable."""
        if v < 10:
            logger.warning(f"Timeout {v} is too short. Setting to 10 seconds.")
            return 10
        elif v > 300:
            logger.warning(f"Timeout {v} is too long. Setting to 300 seconds (5 minutes).")
            return 300
        return v


class TTSConfig(BaseModel):
    """Text-to-speech configuration."""
    provider: str = Field("elevenlabs", description="TTS provider (elevenlabs, openai)")
    api_key: str = Field(default_factory=lambda: os.environ.get("ELEVENLABS_API_KEY", ""))
    voice_id: str = Field("7fbQ7yJuEo56rYjrYaEh", description="Voice ID to use") #male
    # voice_id: str = Field("Xb7hH8MSUJpSbSDYk0k2", description="Voice ID to use") #femaleXb7hH8MSUJpSbSDYk0k2
    speaking_rate: float = Field(1.1, description="Speaking rate multiplier for pacing")
    optimize_for_platform: bool = Field(True, description="Whether to optimize voice for platform")
    voice_style: str = Field("natural", description="Voice style (natural, enthusiastic, serious)")
    model: str = Field("eleven_turbo_v2", description="TTS model to use")

    @field_validator('speaking_rate')
    @classmethod
    def validate_speaking_rate(cls, v):
        """Validate speaking rate is in a reasonable range."""
        if v < 0.5 or v > 2.0:
            logger.warning(f"Speaking rate {v} is outside reasonable range (0.5-2.0). Clamping to valid range.")
            return max(0.5, min(v, 2.0))
        return v


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

    @field_validator('candidates_per_prompt')
    @classmethod
    def validate_candidates(cls, v):
        """Validate number of candidates is reasonable."""
        if v < 1:
            logger.warning("candidates_per_prompt must be at least 1. Setting to 1.")
            return 1
        elif v > 10:
            logger.warning("candidates_per_prompt is too high. Setting to 10.")
            return 10
        return v


class VisualConfig(BaseModel):
    """Visual styling configuration."""
    color_scheme: str = Field("vibrant", description="Color scheme (vibrant, muted, professional, etc.)")
    text_animation: bool = Field(True, description="Whether to animate text")
    motion_effects: bool = Field(True, description="Whether to use motion effects like zooms")
    transition_style: str = Field("smooth", description="Transition style (smooth, sharp, creative)")
    visual_consistency: bool = Field(True, description="Whether to enforce visual consistency")
    apply_filters: bool = Field(True, description="Whether to apply visual filters")
    filter_intensity: float = Field(0.3, description="Intensity of filters (0.0-1.0)")

    @field_validator('filter_intensity')
    @classmethod
    def validate_filter_intensity(cls, v):
        """Validate filter intensity is in valid range."""
        if v < 0.0 or v > 1.0:
            logger.warning(f"Filter intensity {v} is outside valid range (0.0-1.0). Clamping to valid range.")
            return max(0.0, min(v, 1.0))
        return v


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

    @field_validator('quality_threshold')
    @classmethod
    def validate_quality_threshold(cls, v):
        """Validate quality threshold is in valid range."""
        if v < 0.0 or v > 1.0:
            logger.warning(f"Quality threshold {v} is outside valid range (0.0-1.0). Clamping to valid range.")
            return max(0.0, min(v, 1.0))
        return v

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
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in config file {file_path}")
            return cls()
        except Exception as e:
            logger.error(f"Error loading config from {file_path}: {e}")
            return cls()

    def to_file(self, file_path: Union[str, Path]) -> bool:
        """Save configuration to a JSON file."""
        file_path = Path(file_path)
        try:
            # Create parent directory if it doesn't exist
            file_path.parent.mkdir(exist_ok=True, parents=True)

            with open(file_path, "w") as f:
                json.dump(self.model_dump(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving config to {file_path}: {e}")
            return False

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

    def override_from_env(self) -> None:
        """Override configuration values from environment variables."""
        # Override API keys
        if os.environ.get("DEEPSEEK_API_KEY"):
            self.llm.api_key = os.environ.get("DEEPSEEK_API_KEY")

        if os.environ.get("ELEVENLABS_API_KEY"):
            self.tts.api_key = os.environ.get("ELEVENLABS_API_KEY")

        if os.environ.get("STABILITY_API_KEY"):
            self.image_gen.api_key = os.environ.get("STABILITY_API_KEY")

        # Override platform
        if os.environ.get("DEFAULT_PLATFORM") and os.environ.get("DEFAULT_PLATFORM") in PLATFORM_CONFIGS:
            self.platform = os.environ.get("DEFAULT_PLATFORM")
            self.platform_config = PLATFORM_CONFIGS[self.platform]

        # Override model names
        if os.environ.get("LLM_MODEL"):
            self.llm.model = os.environ.get("LLM_MODEL")

        if os.environ.get("TTS_MODEL"):
            self.tts.model = os.environ.get("TTS_MODEL")

        if os.environ.get("IMAGE_GEN_MODEL"):
            self.image_gen.model = os.environ.get("IMAGE_GEN_MODEL")

        # Override timeouts
        if os.environ.get("LLM_TIMEOUT"):
            try:
                self.llm.timeout = int(os.environ.get("LLM_TIMEOUT"))
            except ValueError:
                pass


# Try to load config from default location
config_path = BASE_DIR / "config.json"
DEFAULT_CONFIG = PipelineConfig()

if config_path.exists():
    try:
        DEFAULT_CONFIG = PipelineConfig.from_file(config_path)
    except Exception as e:
        logger.error(f"Error loading default config: {e}")

# Override with environment variables
DEFAULT_CONFIG.override_from_env()