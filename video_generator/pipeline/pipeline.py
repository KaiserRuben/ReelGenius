from typing import Dict, List, Any, Optional, Union, TypedDict, Annotated, Callable
import os
import asyncio
from loguru import logger
import time
import uuid

try:
    # Try importing with new API structure
    from langgraph.graph import StateGraph, END
    from pydantic import BaseModel, Field


    # Define a schema for the state
    class GraphState(TypedDict, total=False):
        """State schema for the LangGraph workflow."""
        content: str  # Original content (removed Annotated since we want to pass it through all steps)
        run_id: str  # Unique ID for this run
        original_content: Optional[str]  # Original content (preserved)
        input_analysis: Optional[Dict[str, Any]]  # Results of content analysis
        is_duplicate: Optional[bool]  # Whether content is duplicate
        content_strategy: Optional[Dict[str, Any]]  # Content strategy

        # Script generation data
        script: Optional[Dict[str, Any]]  # Generated script
        script_quality: Optional[Dict[str, Any]]  # Script quality check results

        # Visual planning data
        visual_plan: Optional[Dict[str, Any]]  # Visual planning data
        visual_validation: Optional[Dict[str, Any]]  # Visual plan validation results

        # Media generation data
        processed_scenes: Optional[List[Dict[str, Any]]]  # Processed media for scenes
        media_success: Optional[bool]  # Whether media generation succeeded
        hook_audio_path: Optional[str]  # Path to hook audio file

        # Final output data
        metadata: Optional[Dict[str, Any]]  # Generated metadata
        video_path: Optional[str]  # Path to the generated video

        # Pipeline control and status
        success: Optional[bool]  # Whether the pipeline succeeded
        error: Optional[str]  # Error message if the pipeline failed
        regeneration_attempts: Optional[int]  # Number of regeneration attempts
        execution_time: Optional[float]  # Execution time in seconds
        progress: Optional[float]  # Progress of the pipeline (0.0 to 1.0)
        progress_callback: Optional[Callable[[float], None]]  # Progress callback function


    # Flag for new API
    NEW_LANGGRAPH_API = True
except ImportError:
    try:
        # Fall back to potential older import structure
        from langgraph.graph import StateGraph

        END = "end"  # In older versions, END might be a string
        NEW_LANGGRAPH_API = False
    except ImportError:
        logger.error("Failed to import LangGraph. Please ensure it's installed correctly.")
        raise

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
        try:
            self.graph = self._build_graph()
            logger.info("LangGraph workflow built successfully")
        except Exception as e:
            logger.error(f"Error building LangGraph workflow: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _build_graph(self):
        """Build the LangGraph workflow with compatibility handling."""

        # New API requires state_schema
        logger.info("Using new LangGraph API with state schema")

        # Define the graph with a state schema
        workflow = StateGraph(state_schema=GraphState)

        # Add nodes
        workflow.add_node("process_input", self._wrap_with_progress(self.input_processor.process, 0.0, 0.1))
        workflow.add_node("generate_script", self._wrap_with_progress(self.script_generator.process, 0.1, 0.3))
        workflow.add_node("plan_visuals", self._wrap_with_progress(self.visual_planner.process, 0.3, 0.5))
        workflow.add_node("media_generation", self._async_wrapper(self._wrap_with_progress(
            self.media_generator.process, 0.5, 0.8)))
        workflow.add_node("assemble_video", self._wrap_with_progress(self.video_assembler.process, 0.8, 1.0))

        # Add edges
        workflow.add_edge("process_input", "generate_script")
        workflow.add_edge("generate_script", "plan_visuals")
        workflow.add_edge("plan_visuals", "media_generation")
        workflow.add_edge("media_generation", "assemble_video")

        # Add conditional edges
        workflow.add_conditional_edges(
            "process_input",
            self._check_duplicate_content,
            {
                "continue": "generate_script",
                "skip": END
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

        # Compile the graph
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
            # Get script quality directly from the state
            script_quality = state.get("script_quality", {})

            if not script_quality.get("passes_quality_check", True):
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

    def _update_progress(self, state: Dict[str, Any], progress: float) -> Dict[str, Any]:
        """Update progress in state and call progress callback if available."""
        state["progress"] = progress

        # Call progress callback if available
        if "progress_callback" in state and callable(state["progress_callback"]):
            try:
                # Check for task_id, which might be needed by some callbacks
                task_id = state.get("task_id")
                if task_id is None:
                    task_id = state.get("run_id")  # Use run_id as fallback

                # Call callback with task_id if available
                if task_id:
                    state["progress_callback"](progress, task_id=task_id)
                else:
                    state["progress_callback"](progress)
            except Exception as e:
                logger.error(f"Error calling progress callback: {e}")

        return state

    def _wrap_with_progress(self, func, start_progress: float, end_progress: float):
        """Wrap a component function with progress tracking."""

        # Check if the function is async
        is_async = asyncio.iscoroutinefunction(func)

        if is_async:
            # Handle async function
            async def async_wrapper(state):
                # Update progress to start value
                state = self._update_progress(state, start_progress)

                # Call the original async function and await its result
                result = await func(state)

                # Update progress to end value
                result = self._update_progress(result, end_progress)

                return result

            return async_wrapper
        else:
            # Handle synchronous function (original implementation)
            def wrapper(state):
                # Update progress to start value
                state = self._update_progress(state, start_progress)

                # Call the original function
                result = func(state)

                # Update progress to end value
                result = self._update_progress(result, end_progress)

                return result

            return wrapper

    # Replace the old _async_wrapper with a better implementation
    def _async_wrapper(self, func):
        """Wrapper to run async functions in LangGraph."""

        def wrapper(state):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(func(state))

        return wrapper

    def run(self, content: str, output_path: Optional[str] = None, task_id: Optional[str] = None,
            progress_callback: Optional[Callable[[float], None]] = None) -> Dict[str, Any]:
        """Run the full video generation pipeline."""
        start_time = time.time()

        # Generate a unique ID for this run if not provided
        run_id = task_id or str(uuid.uuid4())

        try:
            # Set output path if provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Initialize state
            initial_state = {
                "content": content,
                "run_id": run_id,
                "task_id": task_id,  # Store task_id separately for progress callbacks
                "progress": 0.0
            }

            # Add progress callback if provided
            if progress_callback:
                initial_state["progress_callback"] = progress_callback

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

            # Ensure progress is at 100% for completed tasks
            if result.get("success", False):
                result["progress"] = 1.0

                # Final progress callback
                if progress_callback:
                    try:
                        progress_callback(1.0)
                    except Exception as e:
                        logger.error(f"Error in final progress callback: {e}")

            return result

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # Call progress callback with failure
            if progress_callback:
                try:
                    progress_callback(1.0)  # Set to 100% even on failure
                except Exception as e2:
                    logger.error(f"Error in failure progress callback: {e2}")

            return {
                "success": False,
                "error": str(e),
                "run_id": run_id,
                "execution_time": time.time() - start_time,
                "progress": 1.0
            }