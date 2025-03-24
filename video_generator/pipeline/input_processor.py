from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from loguru import logger
import os
import json
import hashlib
from langchain_core.messages import SystemMessage, HumanMessage

from ..models.llm import DeepSeekChatModel, PromptTemplateManager
from ..config import PipelineConfig


class InputType(BaseModel):
    """Detected input type and properties."""
    type: str = Field(..., description="Input type (scientific, concept, marketing, educational, raw)")
    topics: List[str] = Field(default_factory=list, description="Main topics detected")
    complexity: str = Field("medium", description="Content complexity (simple, medium, complex)")
    target_audience: str = Field("general", description="Target audience")
    sentiment: str = Field("neutral", description="Content sentiment")
    key_points: List[str] = Field(default_factory=list, description="Key points extracted")


class ContentStrategy(BaseModel):
    """Platform-optimized content strategy."""
    hook_type: str = Field(..., description="Type of hook (question, fact, challenge, etc.)")
    hook_content: str = Field(..., description="Specific hook content")
    narrative_style: str = Field(..., description="Narrative style (educational, conversational, etc.)")
    engagement_approach: str = Field(..., description="How to drive engagement")
    trending_elements: List[str] = Field(default_factory=list, description="Relevant trending elements")
    hashtags: List[str] = Field(default_factory=list, description="Recommended hashtags")
    optimal_length: int = Field(..., description="Optimal video length in seconds")
    unique_angle: str = Field(..., description="Unique angle for this content")


class InputProcessor:
    """Process and analyze input content for video creation."""

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
        self.history_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "history")
        os.makedirs(self.history_dir, exist_ok=True)

    def analyze_input(self, content: str) -> InputType:
        """Analyze input content to determine type and properties."""
        # Get prompt template
        prompt_data = self.config.get_prompts_data()
        prompt_data["content"] = content

        prompt = self.prompt_manager.render_template("content_analysis", **prompt_data)

        # Use meta-prompting if enabled
        if self.config.llm.use_meta_prompting:
            try:
                prompt = self.prompt_manager.optimize_prompt(
                    prompt,
                    "content analysis",
                    self.config.model_dump()
                )
            except Exception as e:
                logger.warning(f"Meta-prompting failed, using original prompt: {e}")

        output_schema = {
            "type": "string",
            "topics": ["string"],
            "complexity": "string",
            "target_audience": "string",
            "sentiment": "string",
            "key_points": ["string"]
        }

        messages = [
            # SystemMessage(content="You are an expert content analyzer."),
            HumanMessage(content=prompt)
        ]

        try:
            # Log the prompt if detailed logging is enabled
            if self.config.enable_detailed_logging:
                logger.info(f"Content Analysis Prompt:\n{prompt}")

            result = self.llm.structured_output(messages, output_schema)
            return InputType(**result)
        except Exception as e:
            logger.error(f"Error analyzing input: {e}")
            # Provide reasonable defaults if analysis fails
            return InputType(
                type="general",
                topics=["general"],
                complexity="medium",
                target_audience="general",
                sentiment="neutral",
                key_points=["Main content point"]
            )

    def check_topic_history(self, topics: List[str]) -> bool:
        """Check if topics have been covered before to avoid duplication."""
        if not self.config.history_tracking:
            return False  # History tracking disabled

        try:
            # Create a unique hash for the topics
            topics_hash = hashlib.md5(json.dumps(sorted(topics)).encode()).hexdigest()
            history_file = os.path.join(self.history_dir, "topics_history.json")

            # Load history
            if os.path.exists(history_file):
                with open(history_file, "r") as f:
                    history = json.load(f)
            else:
                history = []

            # Check if hash exists
            duplicate = topics_hash in history

            # If not duplicate, add to history
            if not duplicate:
                history.append(topics_hash)
                with open(history_file, "w") as f:
                    json.dump(history, f)

            return duplicate
        except Exception as e:
            logger.error(f"Error checking topic history: {e}")
            return False  # Assume not duplicate on error

    def develop_content_strategy(self, input_type: InputType) -> ContentStrategy:
        """Develop a platform-optimized content strategy based on input analysis."""
        # Get prompt template
        prompt_data = self.config.get_prompts_data()
        prompt_data["input_analysis"] = input_type.model_dump_json(indent=2)

        prompt = self.prompt_manager.render_template("content_strategy", **prompt_data)

        # Use meta-prompting if enabled
        if self.config.llm.use_meta_prompting:
            try:
                prompt = self.prompt_manager.optimize_prompt(
                    prompt,
                    "content strategy development",
                    self.config.model_dump()
                )
            except Exception as e:
                logger.warning(f"Meta-prompting failed for content strategy, using original prompt: {e}")

        output_schema = {
            "hook_type": "string",
            "hook_content": "string",
            "narrative_style": "string",
            "engagement_approach": "string",
            "trending_elements": ["string"],
            "hashtags": ["string"],
            "optimal_length": "integer",
            "unique_angle": "string"
        }

        messages = [
            SystemMessage(content=f"You are a {self.config.platform} content strategy expert."),
            HumanMessage(content=prompt)
        ]

        try:
            # Log the prompt if detailed logging is enabled
            if self.config.enable_detailed_logging:
                logger.info(f"Content Strategy Prompt:\n{prompt}")

            result = self.llm.structured_output(messages, output_schema)

            # Ensure optimal length is within platform constraints
            min_duration = self.config.platform_config.min_duration
            max_duration = self.config.platform_config.max_duration

            if result["optimal_length"] < min_duration:
                result["optimal_length"] = min_duration
            elif result["optimal_length"] > max_duration:
                result["optimal_length"] = max_duration

            return ContentStrategy(**result)
        except Exception as e:
            logger.error(f"Error developing content strategy: {e}")
            # Provide reasonable defaults if strategy development fails
            return ContentStrategy(
                hook_type="question",
                hook_content="Did you know this fascinating fact?",
                narrative_style="conversational",
                engagement_approach="ask for comments",
                trending_elements=["informative content"],
                hashtags=[f"#{self.config.platform}", "learn", "education"],
                optimal_length=self.config.platform_config.min_duration + 15,
                unique_angle="Educational perspective"
            )

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process input content and prepare for video creation."""
        try:
            # Extract content from state
            content = state.get("content", "")
            if not content:
                logger.error("No content provided in state")
                return {
                    "success": False,
                    "error": "No content provided",
                    "run_id": state.get("run_id", "unknown")
                }

            # Store original content for later use
            original_content = content

            # Analyze input
            logger.info("Analyzing input content...")
            input_analysis = self.analyze_input(content)
            logger.info(f"Input analyzed as type: {input_analysis.type} with topics: {input_analysis.topics}")

            # Check history for duplicates
            is_duplicate = self.check_topic_history(input_analysis.topics)
            if is_duplicate:
                logger.warning(f"Topics {input_analysis.topics} have been covered before")

            # Develop content strategy
            logger.info("Developing content strategy...")
            content_strategy = self.develop_content_strategy(input_analysis)
            logger.info(
                f"Developed content strategy with hook type: {content_strategy.hook_type}, optimal length: {content_strategy.optimal_length}s")

            # Return processed input with original content preserved
            output = {
                # Don't modify content field, just preserve it in original_content
                "original_content": original_content,  # Add explicit original_content field
                "input_analysis": input_analysis.model_dump(),
                "content_strategy": content_strategy.model_dump(),
                "is_duplicate": is_duplicate
            }

            # Copy all other state fields to output
            for key, value in state.items():
                if key not in output:
                    output[key] = value
                    
            # Make sure the content field exists
            if "content" not in output and "content" in state:
                output["content"] = state["content"]

            return output

        except Exception as e:
            logger.error(f"Error in input processor: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # Return error state
            return {
                "success": False,
                "error": f"Input processing failed: {str(e)}",
                "run_id": state.get("run_id", "unknown")
            }