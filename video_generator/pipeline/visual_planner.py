
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from loguru import logger
from langchain_core.messages import SystemMessage, HumanMessage

from ..models.llm import DeepSeekChatModel, PromptTemplateManager
from ..config import PipelineConfig

class VisualScene(BaseModel):
    """Visual scene specification for video."""
    segment_index: int = Field(..., description="Index of script segment this visual is for")
    duration: int = Field(..., description="Duration in seconds")
    image_prompt: str = Field(..., description="Detailed image generation prompt")
    text_overlay: Optional[str] = Field(None, description="Text to overlay on image")
    text_position: str = Field("center", description="Position of text overlay")
    effect: Optional[str] = Field(None, description="Visual effect to apply")
    transition: Optional[str] = Field(None, description="Transition to next scene")

class VisualPlan(BaseModel):
    """Complete visual plan for video."""
    scenes: List[VisualScene] = Field(..., description="Visual scenes")
    style_consistency: str = Field(..., description="Visual style to maintain throughout")
    color_palette: List[str] = Field(..., description="Color palette to use")
    text_style: str = Field(..., description="Text style for overlays")

class VisualPlanner:
    """Plan visuals for videos."""
    
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
    
    def generate_visual_plan(self, state: Dict[str, Any]) -> VisualPlan:
        """Generate a visual plan for a video."""
        script = state["script"]
        
        # Get prompt template
        prompt_data = self.config.get_prompts_data()
        prompt_data.update({
            "script": script,
            "input_analysis": state["input_analysis"],
            "content_strategy": state["content_strategy"]
        })
        
        prompt = self.prompt_manager.render_template("visual_planning", **prompt_data)
        
        # Use meta-prompting if enabled
        if self.config.llm.use_meta_prompting:
            prompt = self.prompt_manager.optimize_prompt(
                prompt, 
                "visual planning", 
                self.config.model_dump()
            )
        
        output_schema = {
            "scenes": [
                {
                    "segment_index": "integer",
                    "duration": "integer",
                    "image_prompt": "string",
                    "text_overlay": "string",
                    "text_position": "string",
                    "effect": "string",
                    "transition": "string"
                }
            ],
            "style_consistency": "string",
            "color_palette": ["string"],
            "text_style": "string"
        }
        
        messages = [
            SystemMessage(content=f"You are a {self.config.platform} visual design expert."),
            HumanMessage(content=prompt)
        ]
        
        try:
            # Log the prompt if detailed logging is enabled
            if self.config.enable_detailed_logging:
                logger.info(f"Visual Planning Prompt:\n{prompt}")
                
            result = self.llm.structured_output(messages, output_schema)
            visual_plan = VisualPlan(**result)
            
            # Optimize image prompts
            for scene in visual_plan.scenes:
                # Get optimized image prompt
                image_prompt_data = self.config.get_prompts_data()
                image_prompt_data["base_prompt"] = scene.image_prompt
                
                optimized_prompt = self.prompt_manager.render_template("image_prompt", **image_prompt_data)
                
                # Use meta-prompting if enabled
                if self.config.llm.use_meta_prompting:
                    optimized_prompt = self.prompt_manager.optimize_prompt(
                        optimized_prompt, 
                        "image prompt optimization", 
                        self.config.model_dump()
                    )
                
                # Replace the original prompt with the optimized one
                scene.image_prompt = optimized_prompt
            
            logger.info(f"Generated visual plan with {len(visual_plan.scenes)} scenes")
            return visual_plan
            
        except Exception as e:
            logger.error(f"Error generating visual plan: {e}")
            # Create a basic default visual plan
            default_scenes = []
            
            for i, segment in enumerate(script["segments"]):
                default_scenes.append(
                    VisualScene(
                        segment_index=i,
                        duration=segment["duration"],
                        image_prompt=f"A clear visual representing: {segment['text']}. {self.config.platform_config.aspect_ratio} format, high quality, engaging.",
                        text_overlay=segment["text"][:50] + "..." if len(segment["text"]) > 50 else segment["text"],
                        text_position="center",
                        effect="none",
                        transition="fade"
                    )
                )
            
            return VisualPlan(
                scenes=default_scenes,
                style_consistency="consistent, clear visuals",
                color_palette=["#FFFFFF", "#000000", "#3498DB"],
                text_style="bold, readable font with drop shadow"
            )
    
    def validate_visual_plan(self, visual_plan: VisualPlan, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the visual plan against the script."""
        script = state["script"]
        
        # Basic validation
        script_segments = script["segments"]
        plan_scenes = visual_plan.scenes
        
        # Check that we have a visual for each script segment
        if len(script_segments) != len(plan_scenes):
            logger.warning(f"Mismatch between script segments ({len(script_segments)}) and visual scenes ({len(plan_scenes)})")
            
            # If we have too few scenes, add basic ones
            if len(plan_scenes) < len(script_segments):
                for i in range(len(plan_scenes), len(script_segments)):
                    plan_scenes.append(
                        VisualScene(
                            segment_index=i,
                            duration=script_segments[i]["duration"],
                            image_prompt=f"A visual representing: {script_segments[i]['text']}. {self.config.platform_config.aspect_ratio} format.",
                            text_overlay=script_segments[i]["text"][:50] + "..." if len(script_segments[i]["text"]) > 50 else script_segments[i]["text"],
                            text_position="center",
                            effect="none",
                            transition="fade"
                        )
                    )
            
            # If we have too many scenes, truncate
            elif len(plan_scenes) > len(script_segments):
                plan_scenes = plan_scenes[:len(script_segments)]
        
        # Check that durations align
        for i, (segment, scene) in enumerate(zip(script_segments, plan_scenes)):
            if segment["duration"] != scene.duration:
                logger.warning(f"Duration mismatch for segment {i}: script={segment['duration']}s, visual={scene.duration}s")
                scene.duration = segment["duration"]
        
        # Return validation results
        return {
            "is_valid": True,
            "issues": [],
            "visual_plan": visual_plan.model_dump()
        }

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate and validate visual plan for video."""
        # Generate visual plan using the state data
        logger.info("Generating visual plan...")
        visual_plan = self.generate_visual_plan(state)

        # Validate plan
        logger.info("Validating visual plan...")
        validation = self.validate_visual_plan(visual_plan, state)

        # Update state with visual plan
        result = state.copy()
        result["visual_plan"] = validation["visual_plan"]
        result["visual_validation"] = validation
        
        return result