
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import os
import json
import asyncio

from langgraph.graph.state import CompiledStateGraph
from loguru import logger
import time
import uuid

from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage

from ..config import PipelineConfig, DEFAULT_CONFIG
from .input_processor import InputProcessor
from .script_generator import ScriptGenerator
from .visual_planner import VisualPlanner
from .media_generator import MediaGenerator
from .video_assembler import VideoAssembler

class VideoPipeline:
    """Main pipeline for video generation."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or DEFAULT_CONFIG
        
        # Initialize components
        self.input_processor = InputProcessor(self.config)
        self.script_generator = ScriptGenerator(self.config)
        self.visual_planner = VisualPlanner(self.config)
        self.media_generator = MediaGenerator(self.config)
        self.video_assembler = VideoAssembler(self.config)
        
        # Initialize graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> CompiledStateGraph:
        """Build the LangGraph workflow."""
        # Define the graph
        workflow = StateGraph(name="video_generation")
        
        # Add nodes
        workflow.add_node("process_input", self.input_processor.process)
        workflow.add_node("generate_script", self.script_generator.process)
        workflow.add_node("plan_visuals", self.visual_planner.process)
        workflow.add_node("media_generation", self._async_wrapper(self.media_generator.process))
        workflow.add_node("assemble_video", self.video_assembler.process)
        
        # Add edges
        workflow.add_edge("process_input", "generate_script")
        workflow.add_edge("generate_script", "plan_visuals")
        workflow.add_edge("plan_visuals", "media_generation")
        workflow.add_edge("media_generation", "assemble_video")
        
        # Add conditional edges for quality checks
        workflow.add_conditional_edges(
            "process_input",
            self._check_duplicate_content,
            {
                "continue": "generate_script",
                "skip": "end"
            }
        )
        
        workflow.add_conditional_edges(
            "generate_script",
            self._check_script_quality,
            {
                "continue": "plan_visuals",
                "regenerate": "generate_script"
            }
        )
        
        # Set entry point
        workflow.set_entry_point("process_input")
        
        # Compile
        return workflow.compile()
    
    def _check_duplicate_content(self, state: Dict[str, Any]) -> str:
        """Check if content is duplicate and should be skipped."""
        if state.get("is_duplicate", False) and self.config.history_tracking:
            logger.warning("Duplicate content detected. Skipping.")
            return "skip"
        return "continue"
    
    def _check_script_quality(self, state: Dict[str, Any]) -> str:
        """Check script quality and decide whether to regenerate."""
        try:
            script_data = state.get("script_data", {})
            quality_check = script_data.get("quality_check", {})
            
            if not quality_check.get("passes_quality_check", True):
                # If we've already tried to regenerate, continue anyway
                if state.get("regeneration_attempts", 0) >= self.config.max_retries:
                    logger.warning("Max regeneration attempts reached. Continuing with current script.")
                    return "continue"
                
                # Increment regeneration attempts
                state["regeneration_attempts"] = state.get("regeneration_attempts", 0) + 1
                logger.info(f"Script quality check failed. Regenerating (attempt {state['regeneration_attempts']}).")
                return "regenerate"
            
            return "continue"
        except Exception as e:
            logger.error(f"Error in quality check: {e}")
            return "continue"  # Continue on error
    
    def _async_wrapper(self, async_func):
        """Wrapper to run async functions in LangGraph."""
        def wrapper(state):
            return asyncio.run(async_func(state))
        return wrapper
    
    def run(self, content: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Run the full video generation pipeline."""
        start_time = time.time()
        
        # Generate a unique ID for this run
        run_id = str(uuid.uuid4())
        
        try:
            # Set output path if provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Initialize state
            initial_state = {"content": content, "run_id": run_id}
            
            # Execute graph
            logger.info(f"Starting video generation pipeline with run_id: {run_id}")
            result = self.graph.invoke(initial_state)
            
            execution_time = time.time() - start_time
            logger.info(f"Pipeline completed in {execution_time:.2f} seconds")
            
            # Process result
            if result.get("success", False):
                # If specific output path provided, move video
                if output_path and "video_path" in result:
                    import shutil
                    shutil.copy2(result["video_path"], output_path)
                    result["output_path"] = output_path
                
                logger.info(f"Successfully generated video: {result.get('video_path', 'unknown')}")
            else:
                logger.error(f"Pipeline failed: {result.get('error', 'Unknown error')}")
            
            # Add execution stats
            result["execution_time"] = execution_time
            result["run_id"] = run_id
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                "success": False,
                "error": str(e),
                "run_id": run_id,
                "execution_time": time.time() - start_time
            }


