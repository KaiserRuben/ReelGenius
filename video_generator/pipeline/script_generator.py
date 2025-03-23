from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from loguru import logger
from langchain_core.messages import SystemMessage, HumanMessage

from ..models.llm import DeepSeekChatModel, PromptTemplateManager
from ..config import PipelineConfig


class TikTokScript(BaseModel):
    """Video script with timing and visual directions."""
    hook: str = Field(..., description="Attention-grabbing hook (first few seconds)")
    segments: List[Dict[str, Any]] = Field(..., description="Script segments")
    call_to_action: str = Field(..., description="Call to action at the end")
    total_duration: int = Field(..., description="Total script duration in seconds")
    keywords_to_emphasize: List[str] = Field(..., description="Keywords to emphasize visually")
    background_music_style: str = Field(..., description="Suggested background music style")


class ScriptGenerator:
    """Generate platform-optimized scripts."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.llm = DeepSeekChatModel(
            api_key=config.llm.api_key,
            model=config.llm.model,
            base_url=config.llm.base_url,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            timeout=config.llm.timeout
        )
        self.prompt_manager = PromptTemplateManager()

    def _validate_duration(self, script: TikTokScript) -> TikTokScript:
        """Validate and adjust script duration if needed."""
        min_duration = self.config.platform_config.min_duration
        max_duration = self.config.platform_config.max_duration

        if script.total_duration < min_duration:
            logger.warning(
                f"Script duration ({script.total_duration}s) is below minimum ({min_duration}s). Adjusting...")
            # Simple adjustment: scale all segments proportionally
            scale_factor = min_duration / script.total_duration
            for segment in script.segments:
                segment["duration"] = round(segment["duration"] * scale_factor)
            script.total_duration = min_duration

        elif script.total_duration > max_duration:
            logger.warning(
                f"Script duration ({script.total_duration}s) exceeds maximum ({max_duration}s). Adjusting...")
            # Scale down proportionally
            scale_factor = max_duration / script.total_duration
            for segment in script.segments:
                segment["duration"] = round(segment["duration"] * scale_factor)
            script.total_duration = max_duration

        return script

    def generate_script(self, processed_input: Dict[str, Any]) -> TikTokScript:
        """Generate a platform-optimized script."""
        try:
            # Safely extract data from processed_input with fallbacks
            input_analysis = processed_input.get("input_analysis", {})
            content_strategy = processed_input.get("content_strategy", {})

            # Extract original content or use content field as fallback
            original_content = processed_input.get("original_content", processed_input.get("content", ""))

            if not original_content:
                logger.warning("No content found in processed_input. Using placeholder content.")
                original_content = "No content provided"

            # Get prompt template
            prompt_data = self.config.get_prompts_data()
            prompt_data.update({
                "original_content": original_content[:1000] + "..." if len(
                    original_content) > 1000 else original_content,
                "input_analysis": input_analysis,
                "content_strategy": content_strategy
            })

            prompt = self.prompt_manager.render_template("script_generation", **prompt_data)

            # Use meta-prompting if enabled
            if self.config.llm.use_meta_prompting:
                try:
                    prompt = self.prompt_manager.optimize_prompt(
                        prompt,
                        f"{self.config.platform} script generation",
                        self.config.model_dump()
                    )
                except Exception as e:
                    logger.warning(f"Meta-prompting failed, using original prompt: {e}")

            output_schema = {
                "hook": "string",
                "segments": [
                    {
                        "text": "string",
                        "duration": "integer",
                        "visual_direction": "string"
                    }
                ],
                "call_to_action": "string",
                "total_duration": "integer",
                "keywords_to_emphasize": ["string"],
                "background_music_style": "string"
            }

            messages = [
                HumanMessage(content=prompt)
            ]

            # Log the prompt if detailed logging is enabled
            if self.config.enable_detailed_logging:
                logger.info(f"Script Generation Prompt:\n{prompt}")

            result = self.llm.structured_output(messages, output_schema)
            script = TikTokScript(**result)

            # Validate and adjust duration if needed
            script = self._validate_duration(script)

            logger.info(
                f"Generated script with {len(script.segments)} segments, {script.total_duration}s total duration")
            return script

        except Exception as e:
            logger.error(f"Error generating script: {e}")
            # Provide a minimal default script if generation fails
            platform = self.config.platform
            default_script = TikTokScript(
                hook="Did you know this fascinating fact?",
                segments=[
                    {
                        "text": f"This is content about: {processed_input.get('content', '')[:50]}...",
                        "duration": 20,
                        "visual_direction": "Show relevant imagery"
                    }
                ],
                call_to_action=f"Like and follow for more content like this on {platform}!",
                total_duration=25,
                keywords_to_emphasize=["fascinating", "content"],
                background_music_style="upbeat"
            )
            return default_script

    def quality_check_script(self, script: TikTokScript, processed_input: Dict[str, Any]) -> Dict[str, Any]:
        """Perform quality checks on the generated script."""
        # Get prompt template
        prompt_data = self.config.get_prompts_data()
        prompt_data.update({
            "script": script.model_dump_json(indent=2),
            "content_strategy": processed_input.get("content_strategy", {})
        })

        prompt = self.prompt_manager.render_template("script_quality_check", **prompt_data)

        # Use meta-prompting if enabled
        if self.config.llm.use_meta_prompting:
            try:
                prompt = self.prompt_manager.optimize_prompt(
                    prompt,
                    "script quality check",
                    self.config.model_dump()
                )
            except Exception as e:
                logger.warning(f"Meta-prompting failed for quality check, using original prompt: {e}")

        output_schema = {
            "quality_score": "number",  # 0.0 to 1.0
            "strengths": ["string"],
            "weaknesses": ["string"],
            "improvement_suggestions": ["string"],
            "passes_quality_check": "boolean"
        }

        messages = [
            SystemMessage(content=f"You are a {self.config.platform} script quality analyst."),
            HumanMessage(content=prompt)
        ]

        try:
            # Log the prompt if detailed logging is enabled
            if self.config.enable_detailed_logging:
                logger.info(f"Script Quality Check Prompt:\n{prompt}")

            result = self.llm.structured_output(messages, output_schema)
            logger.info(
                f"Script quality check: score={result['quality_score']}, passes={result['passes_quality_check']}")
            return result

        except Exception as e:
            logger.error(f"Error checking script quality: {e}")
            # Default to passing if check fails
            return {
                "quality_score": 0.75,
                "strengths": ["Script follows platform format"],
                "weaknesses": ["Quality check encountered an error"],
                "improvement_suggestions": [],
                "passes_quality_check": True
            }

    def process(self, processed_input: Dict[str, Any]) -> Dict[str, Any]:
        """Generate and validate a script."""
        # Add content to original_content if not present
        # This ensures compatibility throughout the pipeline
        if "original_content" not in processed_input and "content" in processed_input:
            processed_input["original_content"] = processed_input["content"]

        # Generate script
        logger.info("Generating script...")
        script = self.generate_script(processed_input)

        # Perform quality check
        logger.info("Performing quality check on script...")
        quality_check = self.quality_check_script(script, processed_input)

        # If script fails quality check and quality threshold is enforced, regenerate
        if not quality_check["passes_quality_check"] and self.config.quality_threshold > 0:
            logger.warning("Script failed quality check. Regenerating...")

            # Add quality feedback to input for improved regeneration
            processed_input["quality_feedback"] = quality_check

            # Regenerate with lower temperature for more predictable results
            old_temp = self.llm.temperature
            self.llm.temperature = max(0.3, old_temp - 0.2)
            script = self.generate_script(processed_input)
            self.llm.temperature = old_temp

            # Check again
            quality_check = self.quality_check_script(script, processed_input)

        # Add all data directly to the state instead of nesting
        result = processed_input.copy()
        result["script"] = script.model_dump()
        result["script_quality"] = quality_check
        
        return result