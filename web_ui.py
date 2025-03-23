import os
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = "false"
import streamlit as st
import requests
import json
import time
import os
import logging
from datetime import datetime
from PIL import Image
import base64
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API URL - configurable via environment variable
API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_TIMEOUT = int(os.environ.get("API_TIMEOUT", "10"))  # Default 10 second timeout
SESSION_TIMEOUT = int(os.environ.get("SESSION_TIMEOUT", "3600"))  # Default 1 hour

# Page configuration
st.set_page_config(
    page_title="AI Video Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for styling
st.markdown("""
<style>
.main {
    padding: 2rem;
}
.title {
    font-size: 2.5rem;
    margin-bottom: 1rem;
}
.subtitle {
    font-size: 1.2rem;
    color: #666;
    margin-bottom: 2rem;
}
.stButton>button {
    width: 100%;
}
.video-container {
    margin-top: 2rem;
}
.success {
    color: #0f5132;
    background-color: #d1e7dd;
    border-color: #badbcc;
    padding: 1rem;
    border-radius: 0.25rem;
    margin-bottom: 1rem;
}
.warning {
    color: #664d03;
    background-color: #fff3cd;
    border-color: #ffecb5;
    padding: 1rem;
    border-radius: 0.25rem;
    margin-bottom: 1rem;
}
.error {
    color: #842029;
    background-color: #f8d7da;
    border-color: #f5c2c7;
    padding: 1rem;
    border-radius: 0.25rem;
    margin-bottom: 1rem;
}
.stage-indicator {
    font-size: 1.1rem;
    font-weight: bold;
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
}
.info-box {
    background-color: #cfe2ff;
    padding: 0.75rem;
    border-radius: 0.25rem;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# Session timeout handling
if 'last_activity' not in st.session_state:
    st.session_state.last_activity = time.time()

if time.time() - st.session_state.last_activity > SESSION_TIMEOUT:
    # Reset session
    for key in list(st.session_state.keys()):
        if key != 'last_activity':
            del st.session_state[key]
    st.warning("Your session has expired. Please refresh the page.")

# Update last activity timestamp
st.session_state.last_activity = time.time()

# Initialize session state
if 'tasks' not in st.session_state:
    st.session_state.tasks = []
if 'history' not in st.session_state:
    st.session_state.history = []
if 'metadata_expanded' not in st.session_state:
    st.session_state.metadata_expanded = False
if 'live_monitor' not in st.session_state:
    st.session_state.live_monitor = False

# App title
st.markdown("<h1 class='title'>🎬 AI Video Generator</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Transform content into engaging videos with AI</p>", unsafe_allow_html=True)

# API status indicator
with st.sidebar:
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        if response.status_code == 200:
            status_data = response.json()
            if status_data.get("status") == "healthy":
                st.sidebar.success("✅ API Connected")

                # Show additional API stats
                with st.expander("API Status Details"):
                    st.write(f"Version: {status_data.get('api_version', 'Unknown')}")
                    st.write(f"Platform: {status_data.get('config', {}).get('platform', 'Unknown')}")

                    # Task counts
                    task_counts = status_data.get("tasks", {})
                    st.write("Active Tasks:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"- Running: {task_counts.get('running', 0)}")
                        st.write(f"- Queued: {task_counts.get('queued', 0)}")
                    with col2:
                        st.write(f"- Completed: {task_counts.get('completed', 0)}")
                        st.write(f"- Failed: {task_counts.get('failed', 0)}")

                    # API key status
                    if status_data.get("api_keys_configured", True):
                        st.write("✅ API Keys: Configured")
                    else:
                        st.warning("⚠️ API Keys: Missing")
            else:
                st.sidebar.warning(f"⚠️ API Status: {status_data.get('status', 'degraded')}")
        else:
            st.sidebar.error(f"❌ API Error: {response.status_code}")
    except Exception as e:
        st.sidebar.error(f"❌ API Unavailable: {str(e)}")

# Sidebar
st.sidebar.header("Configuration")

# Platform selection
platform = st.sidebar.selectbox(
    "Select Platform",
    ["tiktok", "youtube_shorts", "instagram_reels", "general"],
    help="Choose the platform you're creating content for. Each has different optimal formats."
)


# Function to get platforms with caching
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_platforms():
    try:
        response = requests.get(f"{API_URL}/platforms", timeout=API_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            # Log the actual structure for debugging
            logger.info(f"Platforms data structure: {json.dumps(data, indent=2)[:500]}...")
            return data
        else:
            logger.error(f"Error fetching platforms: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"Exception fetching platforms: {str(e)}")
        return None


# Platform info expander
with st.sidebar.expander("📊 Platform Information"):
    platforms_data = get_platforms()
    if platforms_data:
        # Log what we got to debug the structure
        st.write(f"Available platforms: {', '.join(platforms_data.get('platforms', []))}")

        # Check if 'configs' exists and has our platform
        if "configs" in platforms_data and platform in platforms_data["configs"]:
            platform_config = platforms_data["configs"][platform]
            st.write(f"**{platform.replace('_', ' ').title()}**")
            st.write(f"📏 Aspect Ratio: {platform_config.get('aspect_ratio', 'N/A')}")
            st.write(
                f"⏱️ Duration: {platform_config.get('min_duration', 'N/A')} - {platform_config.get('max_duration', 'N/A')} seconds")
            st.write(f"🖥️ Resolution: {platform_config.get('resolution', 'N/A')}")
            st.write(f"🪝 Hook Duration: {platform_config.get('hook_duration', 'N/A')} seconds")
        else:
            st.write("Platform configuration not found. Using default settings.")
    else:
        st.write("Could not fetch platform information. API may be unavailable.")

# Advanced settings expander
with st.sidebar.expander("⚙️ Advanced Settings"):
    # Visual settings
    st.subheader("🎨 Visual Settings")
    color_scheme = st.selectbox("Color Scheme", ["vibrant", "muted", "professional", "dark", "light"])
    text_animation = st.checkbox("Text Animation", value=True)
    motion_effects = st.checkbox("Motion Effects (Ken Burns)", value=True)
    transition_style = st.selectbox("Transitions", ["smooth", "sharp", "creative"])

    # Audio settings
    st.subheader("🔊 Audio Settings")
    voice_style = st.selectbox("Voice Style", ["natural", "enthusiastic", "serious"])
    speaking_rate = st.slider("Speaking Rate", min_value=0.8, max_value=1.3, value=1.1, step=0.1)

    # Image settings
    st.subheader("🖼️ Image Settings")
    image_style = st.selectbox("Image Style", ["photorealistic", "3d_render", "cartoon", "sketch", "painting"])
    candidates_per_prompt = st.slider("Images Per Scene", min_value=1, max_value=5, value=3, step=1)

    # Model settings
    st.subheader("🧠 Model Settings")
    use_meta_prompting = st.checkbox("Use Meta-Prompting", value=True,
                                     help="Optimize prompts with a meta-prompt before sending to LLM")
    chain_of_thought = st.checkbox("Chain of Thought", value=True,
                                   help="Encourage step-by-step reasoning in LLM")
    few_shot_examples = st.checkbox("Few-Shot Examples", value=True,
                                    help="Include examples to guide the model")


# Function to check task status with improved error handling
def check_task_status(task_id):
    """Check task status with improved error handling and timeout"""
    try:
        response = requests.get(f"{API_URL}/status/{task_id}", timeout=API_TIMEOUT)

        if response.status_code == 200:
            status_data = response.json()

            # Add basic validation of the response
            if not isinstance(status_data, dict):
                st.error(f"Invalid response format from API: {status_data}")
                return None

            # Check for required fields
            if "status" not in status_data:
                st.error("Invalid status response: missing 'status' field")
                return None

            # Process any error fields
            if status_data.get("status") == "failed" and "error" in status_data:
                # Log error for debugging
                logger.error(f"Task {task_id} failed: {status_data['error']}")

            return status_data

        elif response.status_code == 404:
            st.error(f"Task {task_id} not found")
            return None

        else:
            st.error(f"API error: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.Timeout:
        st.warning("Request timed out. The server might be under heavy load.")
        return {"status": "unknown", "progress": 0, "error": "Request timed out"}

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to API server. Please check if it's running.")
        return None

    except Exception as e:
        st.error(f"Unexpected error: {e}")
        logger.error(f"Error checking task status: {str(e)}")
        return None


# Function to display video with improved handling
def display_video(video_url):
    try:
        # If video is on local filesystem
        if os.path.exists(video_url):
            video_file = open(video_url, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
        # If video is from remote API
        elif video_url.startswith("http"):
            st.video(video_url)
        else:
            st.error(f"Video file not found: {video_url}")
    except Exception as e:
        st.error(f"Error displaying video: {e}")
        logger.error(f"Error displaying video {video_url}: {str(e)}")


# Function to get task history with caching
@st.cache_data(ttl=60)  # Cache for 1 minute
def get_task_history():
    try:
        response = requests.get(f"{API_URL}/tasks", timeout=API_TIMEOUT)
        if response.status_code == 200:
            return response.json()["tasks"]
        else:
            logger.error(f"Error fetching task history: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        logger.error(f"Exception fetching task history: {str(e)}")
        return []


# Main area
tab1, tab2, tab3 = st.tabs(["Create Video", "View History", "Analytics"])

# Create Video Tab
with tab1:
    # Input section
    st.header("Input Content")

    input_type = st.radio("Input Type", ["Text", "File Upload"], horizontal=True)

    if input_type == "Text":
        content = st.text_area("Enter your content here", height=200,
                               placeholder="Paste your article, script, or any content you want to transform into a video...")
    else:
        uploaded_file = st.file_uploader("Upload a text file", type=["txt", "md"])
        if uploaded_file is not None:
            content = uploaded_file.getvalue().decode("utf-8")
            st.text_area("File Content", content, height=200)
        else:
            content = None

    # Config overrides based on advanced settings
    config_overrides = {
        "visual": {
            "color_scheme": color_scheme,
            "text_animation": text_animation,
            "motion_effects": motion_effects,
            "transition_style": transition_style
        },
        "tts": {
            "voice_style": voice_style,
            "speaking_rate": speaking_rate
        },
        "image_gen": {
            "style": image_style,
            "candidates_per_prompt": candidates_per_prompt
        },
        "llm": {
            "use_meta_prompting": use_meta_prompting,
            "chain_of_thought": chain_of_thought,
            "few_shot_examples": few_shot_examples
        }
    }

    col1, col2 = st.columns(2)

    # Generate button
    with col1:
        if st.button("🚀 Generate Video", type="primary"):
            if not content:
                st.error("Please enter or upload content first.")
            else:
                with st.spinner("Submitting video generation request..."):
                    try:
                        # Submit generation request
                        response = requests.post(
                            f"{API_URL}/generate",
                            json={
                                "content": content,
                                "platform": platform,
                                "config_overrides": config_overrides
                            },
                            timeout=API_TIMEOUT
                        )

                        if response.status_code == 200:
                            result = response.json()
                            task_id = result["task_id"]

                            # Store task ID in session state
                            if "tasks" not in st.session_state:
                                st.session_state.tasks = []

                            st.session_state.tasks.append({
                                "id": task_id,
                                "platform": platform,
                                "timestamp": time.time(),
                                "status": "queued"
                            })

                            st.success(f"Video generation started! Task ID: {task_id}")

                            # Switch to task monitoring
                            st.session_state.current_task = task_id
                            st.session_state.live_monitor = True
                            st.rerun()
                        else:
                            st.error(f"API error: {response.status_code} - {response.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("Could not connect to API server. Please check if it's running.")
                    except requests.exceptions.Timeout:
                        st.error("Request timed out. The server might be under heavy load.")
                    except Exception as e:
                        st.error(f"Error submitting generation request: {e}")
                        logger.error(f"Error submitting generation request: {str(e)}")

    # Example button
    with col2:
        if st.button("📝 Use Example Content"):
            example_content = """
            # Quantum Computing Explained

            Quantum computing leverages the principles of quantum mechanics to process information in ways that classical computers cannot. Unlike traditional bits that exist in a state of either 0 or 1, quantum bits or "qubits" can exist in multiple states simultaneously through a property called superposition.

            Another key quantum property is entanglement, where qubits become interconnected and the state of one qubit instantaneously affects another, regardless of distance. This allows quantum computers to perform certain calculations exponentially faster than classical computers.

            Potential applications include:
            - Breaking current encryption methods
            - Drug discovery and molecular modeling
            - Optimization problems in logistics and finance
            - Advanced AI and machine learning

            While still in early stages, quantum computing promises to revolutionize computing as we know it.
            """
            # Set the example content
            st.session_state.example_content = example_content
            # Rerun to update the UI
            st.rerun()

    # Use example content if set
    if "example_content" in st.session_state:
        content = st.session_state.example_content
        # Clean up after use
        del st.session_state.example_content

    # Task monitoring section with improved error handling and status display
    if "current_task" in st.session_state and st.session_state.live_monitor:
        st.header("Current Task Progress")

        task_id = st.session_state.current_task
        status_container = st.container()

        with status_container:
            status_info = check_task_status(task_id)

            if not status_info:
                st.error("Unable to get task status. The server might be unavailable.")
                if st.button("Retry"):
                    st.rerun()
                # Don't use return here - it's outside a function!
                st.stop()  # Use st.stop() instead to halt execution

            status = status_info["status"]
            progress = status_info.get("progress", 0)

            # Show any errors from the task
            if status == "failed" and "error" in status_info:
                st.error(f"Error: {status_info['error']}")

                # Show detailed error information if available
                if "result" in status_info and "error" in status_info["result"]:
                    with st.expander("Detailed Error Information"):
                        st.code(status_info["result"]["error"])

                # Option to start over
                if st.button("Start Over"):
                    st.session_state.live_monitor = False
                    st.rerun()
                st.stop()  # Use st.stop() instead of return

            # Progress bar with better visual feedback
            col1, col2 = st.columns([3, 1])
            with col1:
                progress_bar = st.progress(progress)
            with col2:
                st.metric("Progress", f"{int(progress * 100)}%")

            # Status message with icon
            if status == "queued":
                st.markdown("<div class='info-box'>⏳ Task is queued and waiting to start</div>", unsafe_allow_html=True)
            elif status == "running":
                if progress < 0.25:
                    st.markdown("<div class='info-box stage-indicator'>🔍 Analyzing content...</div>",
                                unsafe_allow_html=True)
                elif progress < 0.5:
                    st.markdown("<div class='info-box stage-indicator'>✍️ Generating script...</div>",
                                unsafe_allow_html=True)
                elif progress < 0.75:
                    st.markdown("<div class='info-box stage-indicator'>🎨 Creating visuals...</div>",
                                unsafe_allow_html=True)
                else:
                    st.markdown("<div class='info-box stage-indicator'>🎬 Assembling video...</div>",
                                unsafe_allow_html=True)

            # Auto-refresh with better user feedback
            if status not in ["completed", "failed"]:
                refresh_message = st.empty()
                for i in range(3, 0, -1):
                    refresh_message.info(f"Refreshing in {i} seconds...")
                    time.sleep(1)
                st.rerun()

            # If task is completed, show the video
            if status == "completed" and "result" in status_info and "video_path" in status_info["result"]:
                progress_bar.progress(1.0)
                st.success("✅ Video generation completed successfully!")

                # Display video
                video_url = status_info["result"]["video_path"]
                st.subheader("Generated Video")
                display_video(video_url)

                # Metadata expander
                with st.expander("Video Metadata", expanded=st.session_state.metadata_expanded):
                    if "metadata" in status_info["result"]:
                        metadata = status_info["result"]["metadata"]

                        # Display in a nice format
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Title:**", metadata.get("title", "N/A"))
                            st.write("**Platform:**", metadata.get("platform", "N/A"))
                            st.write("**Duration:**", f"{metadata.get('duration', 0):.2f} seconds")

                        with col2:
                            st.write("**Resolution:**", metadata.get("resolution", "N/A"))
                            st.write("**Category:**", metadata.get("category", "N/A"))

                        # Description and hashtags
                        st.subheader("Description")
                        st.text_area("", metadata.get("description", ""), height=100, disabled=True)

                        st.subheader("Hashtags")
                        if "hashtags" in metadata:
                            st.write(" ".join(metadata["hashtags"]))
                    else:
                        st.write("No metadata available")

                # Download button
                if os.path.exists(video_url):
                    with open(video_url, "rb") as file:
                        video_bytes = file.read()
                        st.download_button(
                            label="📥 Download Video",
                            data=video_bytes,
                            file_name=os.path.basename(video_url),
                            mime="video/mp4"
                        )
                elif video_url.startswith("http"):
                    st.markdown(f"[📥 Download Video]({video_url})")
                else:
                    st.error("Video file not accessible for download")

                # Option to create another video
                if st.button("Create Another Video"):
                    st.session_state.live_monitor = False
                    st.rerun()

# View History Tab
with tab2:
    st.header("Generation History")

    # Refresh button
    if st.button("🔄 Refresh History"):
        # Force refresh cache
        get_task_history.clear()
        st.session_state.history = get_task_history()
        st.success("History refreshed")

    # Get tasks from cache if not explicitly refreshed
    if not st.session_state.history:
        st.session_state.history = get_task_history()

    # Display tasks with improved information
    if st.session_state.history:
        # Create a detailed table
        task_data = []
        for task in st.session_state.history:
            # Extract creation time and format it nicely
            created_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(task['created_at']))

            # Calculate how long ago
            time_ago = time.time() - task['created_at']
            if time_ago < 60:
                time_ago_str = f"{int(time_ago)}s ago"
            elif time_ago < 3600:
                time_ago_str = f"{int(time_ago / 60)}m ago"
            else:
                time_ago_str = f"{int(time_ago / 3600)}h ago"

            # Add more helpful information like execution time
            execution_time = task.get('execution_time', None)
            if execution_time:
                exec_time_str = f"{execution_time:.1f}s"
            else:
                exec_time_str = "N/A"

            # Add error summary if failed
            error_summary = ""
            if task.get('status') == 'failed' and 'error' in task:
                error_text = task['error']
                error_summary = error_text[:30] + "..." if len(error_text) > 30 else error_text

            task_data.append({
                "ID": task['task_id'][:8] + "...",
                "Status": task['status'],
                "Created": f"{created_time} ({time_ago_str})",
                "Platform": task.get('platform', 'unknown'),
                "Exec Time": exec_time_str,
                "Error": error_summary,
                "Task ID": task['task_id']  # Hidden column for reference
            })

        df = pd.DataFrame(task_data)

        # Add ability to sort by different columns
        sort_by = st.selectbox("Sort by", ["Created", "Status", "Platform", "Exec Time"], index=0)
        ascending = st.checkbox("Ascending order", value=False)

        if sort_by == "Created":
            df = df.sort_values(by=["Created"], ascending=ascending)
        else:
            df = df.sort_values(by=[sort_by], ascending=ascending)


        # Color-code statuses for better visibility
        def highlight_status(val):
            color_map = {
                'completed': 'background-color: #d1e7dd',
                'failed': 'background-color: #f8d7da',
                'running': 'background-color: #cfe2ff',
                'queued': 'background-color: #fff3cd'
            }
            return color_map.get(val, '')


        # Show table with styled status column
        st.dataframe(
            df.style.applymap(highlight_status, subset=['Status']),
            column_config={
                "ID": st.column_config.TextColumn("ID", width="small"),
                "Status": st.column_config.TextColumn("Status", width="small"),
                "Created": st.column_config.TextColumn("Created", width="medium"),
                "Platform": st.column_config.TextColumn("Platform", width="small"),
                "Exec Time": st.column_config.TextColumn("Exec Time", width="small"),
                "Error": st.column_config.TextColumn("Error", width="medium"),
                "Task ID": st.column_config.TextColumn("Task ID", width=None),
            },
            hide_index=True
        )

        # Pagination for task list
        page_size = 10
        if len(st.session_state.history) > page_size:
            num_pages = (len(st.session_state.history) + page_size - 1) // page_size
            current_page = st.selectbox("Page", range(1, num_pages + 1), 1)
            start_idx = (current_page - 1) * page_size
            end_idx = min(start_idx + page_size, len(st.session_state.history))
            paged_tasks = st.session_state.history[start_idx:end_idx]
            task_ids = [task['task_id'] for task in paged_tasks]
        else:
            task_ids = [task['task_id'] for task in st.session_state.history]

        # Task selection
        selected_task = st.selectbox(
            "Select a task to view details",
            options=task_ids,
            format_func=lambda
                x: f"{x[:8]}... ({next((task['status'] for task in st.session_state.history if task['task_id'] == x), 'unknown')})"
        )

        if selected_task:
            # View selected task
            status_info = check_task_status(selected_task)

            if status_info:
                st.subheader("Task Details")

                # Status information
                status = status_info["status"]
                progress = status_info.get("progress", 0)

                # Status indicator
                if status == "completed":
                    st.markdown(f"<div class='success'>Status: {status}</div>", unsafe_allow_html=True)
                elif status == "failed":
                    st.markdown(f"<div class='error'>Status: {status}</div>", unsafe_allow_html=True)
                elif status == "running":
                    st.markdown(f"<div class='info-box'>Status: {status} ({int(progress * 100)}%)</div>",
                                unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='warning'>Status: {status}</div>", unsafe_allow_html=True)

                # Show error if present
                if status == "failed" and "error" in status_info:
                    with st.expander("Error Details"):
                        st.error(status_info.get("error", "Unknown error"))

                        # Show more detailed error info if available
                        if "result" in status_info and isinstance(status_info["result"], dict) and "error" in \
                                status_info["result"]:
                            st.code(status_info["result"]["error"])

                # If completed, show the video
                if status == "completed" and "result" in status_info and "video_path" in status_info["result"]:
                    video_url = status_info["result"]["video_path"]

                    # Display video
                    st.subheader("Generated Video")
                    display_video(video_url)

                    # Download button
                    if os.path.exists(video_url):
                        with open(video_url, "rb") as file:
                            video_bytes = file.read()
                            st.download_button(
                                label="📥 Download Video",
                                data=video_bytes,
                                file_name=os.path.basename(video_url),
                                mime="video/mp4"
                            )
                    elif video_url.startswith("http"):
                        st.markdown(f"[📥 Download Video]({video_url})")
                    else:
                        st.error("Video file not accessible for download")

                    # Metadata expander
                    with st.expander("Video Metadata"):
                        if "metadata" in status_info["result"]:
                            metadata = status_info["result"]["metadata"]
                            st.json(metadata)

                # If running, show progress
                elif status == "running":
                    st.progress(progress)
                    st.info(f"Video generation is in progress ({int(progress * 100)}% completed)")

                    # Offer to view live progress
                    if st.button("Monitor Progress"):
                        st.session_state.current_task = selected_task
                        st.session_state.live_monitor = True
                        st.rerun()

                # Delete button with confirmation
                delete_button = st.button(f"🗑️ Delete Task", key=f"delete_{selected_task}")
                if delete_button:
                    confirm_col1, confirm_col2 = st.columns(2)
                    with confirm_col1:
                        cancel = st.button("Cancel", key=f"cancel_delete_{selected_task}")
                    with confirm_col2:
                        confirm = st.button("⚠️ Confirm Deletion", key=f"confirm_delete_{selected_task}")

                    if confirm:
                        try:
                            response = requests.delete(f"{API_URL}/task/{selected_task}", timeout=API_TIMEOUT)
                            if response.status_code == 200:
                                st.success("Task deleted successfully")
                                # Remove from session state
                                st.session_state.history = [task for task in st.session_state.history if
                                                            task['task_id'] != selected_task]
                                # Force refresh cache
                                get_task_history.clear()
                                st.rerun()
                            else:
                                st.error(f"Error deleting task: {response.status_code} - {response.text}")
                        except Exception as e:
                            st.error(f"Error deleting task: {e}")
                            logger.error(f"Error deleting task: {str(e)}")
    else:
        st.info("No tasks in history. Generate a video or click Refresh to update.")

# Analytics Tab
with tab3:
    st.header("Generation Analytics")

    try:
        # Fetch tasks for analytics
        analytics_tasks = get_task_history()

        if analytics_tasks:
            # Process data for analytics
            platform_counts = {}
            status_counts = {}
            creation_dates = []

            for task in analytics_tasks:
                # Platform stats
                platform = task.get('platform', 'unknown')
                platform_counts[platform] = platform_counts.get(platform, 0) + 1

                # Status stats
                status = task.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1

                # Creation date
                date = time.strftime('%Y-%m-%d', time.localtime(task['created_at']))
                creation_dates.append(date)

            # Create dataframes
            platform_df = pd.DataFrame({
                'Platform': list(platform_counts.keys()),
                'Count': list(platform_counts.values())
            })

            status_df = pd.DataFrame({
                'Status': list(status_counts.keys()),
                'Count': list(status_counts.values())
            })

            date_df = pd.DataFrame({
                'Date': creation_dates
            })
            date_counts = date_df['Date'].value_counts().reset_index()
            date_counts.columns = ['Date', 'Count']
            date_counts = date_counts.sort_values('Date')

            # Display analytics
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Videos by Platform")
                fig1 = px.pie(platform_df, values='Count', names='Platform',
                              hole=.3, color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig1)

            with col2:
                st.subheader("Videos by Status")
                fig2 = px.pie(status_df, values='Count', names='Status',
                              hole=.3, color_discrete_sequence=px.colors.qualitative.Bold)
                st.plotly_chart(fig2)

            # Timeline chart
            st.subheader("Generation Timeline")
            fig3 = px.bar(date_counts, x='Date', y='Count',
                          labels={'Count': 'Number of Videos', 'Date': 'Generation Date'},
                          color_discrete_sequence=['#3498db'])
            st.plotly_chart(fig3)

            # Overall statistics
            st.subheader("Overall Statistics")
            total_videos = len(analytics_tasks)
            completed_videos = status_counts.get('completed', 0)
            completion_rate = (completed_videos / total_videos) * 100 if total_videos > 0 else 0

            stats_col1, stats_col2, stats_col3 = st.columns(3)
            with stats_col1:
                st.metric("Total Videos", total_videos)
            with stats_col2:
                st.metric("Completed Videos", completed_videos)
            with stats_col3:
                st.metric("Completion Rate", f"{completion_rate:.1f}%")

            # Performance analytics
            st.subheader("Performance Analysis")

            # Filter completed tasks with execution time
            completed_tasks = [task for task in analytics_tasks
                               if task.get('status') == 'completed' and 'execution_time' in task]

            if completed_tasks:
                # Extract execution times
                execution_times = [task.get('execution_time', 0) for task in completed_tasks]

                # Create execution time histogram
                fig4 = px.histogram(
                    x=execution_times,
                    nbins=10,
                    labels={'x': 'Execution Time (seconds)', 'y': 'Number of Videos'},
                    title="Video Generation Time Distribution",
                    color_discrete_sequence=['#9b59b6']
                )
                st.plotly_chart(fig4)

                # Calculate performance statistics
                avg_time = sum(execution_times) / len(execution_times)
                median_time = sorted(execution_times)[len(execution_times) // 2]
                max_time = max(execution_times)
                min_time = min(execution_times)

                # Display performance metrics
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                with metrics_col1:
                    st.metric("Average Time", f"{avg_time:.1f}s")
                with metrics_col2:
                    st.metric("Median Time", f"{median_time:.1f}s")
                with metrics_col3:
                    st.metric("Min Time", f"{min_time:.1f}s")
                with metrics_col4:
                    st.metric("Max Time", f"{max_time:.1f}s")
            else:
                st.info("No completed videos with timing data available yet.")

        else:
            st.info("No analytics data available yet. Generate videos to see analytics.")
    except Exception as e:
        st.error(f"Error generating analytics: {e}")
        logger.error(f"Error generating analytics: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center">
        <p>AI Video Generator | Made with ❤️ using Streamlit | API: {API_URL}</p>
    </div>
    """,
    unsafe_allow_html=True
)