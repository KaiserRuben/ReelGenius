# streamlit_app.py
import streamlit as st
import requests
import json
import time
import os
from PIL import Image
import base64
import pandas as pd
import plotly.express as px

# API URL
API_URL = "http://localhost:8000"  # Change this if your API is hosted elsewhere

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
</style>
""", unsafe_allow_html=True)

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

# Sidebar
st.sidebar.header("Configuration")

# Platform selection
platform = st.sidebar.selectbox(
    "Select Platform",
    ["tiktok", "youtube_shorts", "instagram_reels", "general"],
    help="Choose the platform you're creating content for. Each has different optimal formats."
)

# Platform info expander
with st.sidebar.expander("📊 Platform Information"):
    try:
        response = requests.get(f"{API_URL}/platforms")
        if response.status_code == 200:
            platforms_data = response.json()
            if platform in platforms_data["configs"]:
                platform_config = platforms_data["configs"][platform]
                st.write(f"**{platform.replace('_', ' ').title()}**")
                st.write(f"📏 Aspect Ratio: {platform_config.get('aspect_ratio', 'N/A')}")
                st.write(
                    f"⏱️ Duration: {platform_config.get('min_duration', 'N/A')} - {platform_config.get('max_duration', 'N/A')} seconds")
                st.write(f"🖥️ Resolution: {platform_config.get('resolution', 'N/A')}")
                st.write(f"🪝 Hook Duration: {platform_config.get('hook_duration', 'N/A')} seconds")
    except:
        st.write("Could not fetch platform information")

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


# Function to check task status
def check_task_status(task_id):
    try:
        response = requests.get(f"{API_URL}/status/{task_id}")
        return response.json()
    except Exception as e:
        st.error(f"Error checking task status: {e}")
        return None


# Function to display video
def display_video(video_url):
    try:
        video_file = open(video_url, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
    except Exception as e:
        st.error(f"Error displaying video: {e}")


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
                            }
                        )

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

                    except Exception as e:
                        st.error(f"Error submitting generation request: {e}")

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
            st.experimental_rerun()

    # Use example content if set
    if "example_content" in st.session_state:
        content = st.session_state.example_content
        # Clean up after use
        del st.session_state.example_content

    # Task monitoring section
    if "current_task" in st.session_state and st.session_state.live_monitor:
        st.header("Current Task Progress")

        task_id = st.session_state.current_task
        status_info = check_task_status(task_id)

        if status_info:
            status = status_info["status"]
            progress = status_info.get("progress", 0)

            # Progress bar
            progress_bar = st.progress(progress)
            status_text = st.empty()
            status_text.write(f"Status: {status} ({int(progress * 100)}%)")

            # Auto-refresh for live monitoring
            if status not in ["completed", "failed"]:
                time.sleep(2)  # Wait a bit before refreshing
                st.experimental_rerun()

            # If task is completed, show the video
            if status == "completed" and "result" in status_info and "video_path" in status_info["result"]:
                progress_bar.progress(1.0)
                status_text.success("Video generation completed successfully!")

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
                with open(video_url, "rb") as file:
                    video_bytes = file.read()
                    st.download_button(
                        label="📥 Download Video",
                        data=video_bytes,
                        file_name=os.path.basename(video_url),
                        mime="video/mp4"
                    )

                # Stop live monitoring now that we've displayed everything
                st.session_state.live_monitor = False

            # If task failed, show error
            elif status == "failed":
                st.error(f"Video generation failed: {status_info.get('error', 'Unknown error')}")
                st.session_state.live_monitor = False

# View History Tab
with tab2:
    st.header("Generation History")

    # Refresh button
    if st.button("🔄 Refresh History"):
        try:
            response = requests.get(f"{API_URL}/tasks")
            tasks = response.json()["tasks"]

            # Update session state
            st.session_state.history = tasks
        except Exception as e:
            st.error(f"Error fetching task history: {e}")

    # Display tasks
    if st.session_state.history:
        # Create a nice table
        task_data = []
        for task in st.session_state.history:
            task_data.append({
                "ID": task['task_id'][:8] + "...",
                "Status": task['status'],
                "Created": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(task['created_at'])),
                "Platform": task.get('platform', 'unknown'),
                "Task ID": task['task_id']  # Hidden column for reference
            })

        df = pd.DataFrame(task_data)


        # Color-code statuses
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
                "Platform": st.column_config.TextColumn("Platform", width="medium"),
                "Task ID": st.column_config.TextColumn("Task ID", width=None),
            },
            hide_index=True
        )

        # Task selection
        selected_task = st.selectbox(
            "Select a task to view details",
            options=df["Task ID"].tolist(),
            format_func=lambda x: f"{x[:8]}... ({dict(zip(df['Task ID'], df['Status']))[x]})"
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
                    st.markdown(f"<div class='info'>Status: {status} ({int(progress * 100)}%)</div>",
                                unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='warning'>Status: {status}</div>", unsafe_allow_html=True)

                # If completed, show the video
                if status == "completed" and "result" in status_info and "video_path" in status_info["result"]:
                    video_url = status_info["result"]["video_path"]

                    # Display video
                    st.subheader("Generated Video")
                    display_video(video_url)

                    # Download button
                    with open(video_url, "rb") as file:
                        video_bytes = file.read()
                        st.download_button(
                            label="📥 Download Video",
                            data=video_bytes,
                            file_name=os.path.basename(video_url),
                            mime="video/mp4"
                        )

                    # Metadata expander
                    with st.expander("Video Metadata"):
                        if "metadata" in status_info["result"]:
                            metadata = status_info["result"]["metadata"]
                            st.json(metadata)

                # If failed, show error
                elif status == "failed":
                    st.error(f"Error: {status_info.get('error', 'Unknown error')}")

                # Delete button
                if st.button(f"🗑️ Delete Task", key=f"delete_{selected_task}"):
                    try:
                        response = requests.delete(f"{API_URL}/task/{selected_task}")
                        if response.status_code == 200:
                            st.success("Task deleted successfully")
                            # Remove from session state
                            st.session_state.history = [task for task in st.session_state.history if
                                                        task['task_id'] != selected_task]
                            st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error deleting task: {e}")
    else:
        st.info("No tasks in history. Generate a video or click Refresh to update.")

# Analytics Tab
with tab3:
    st.header("Generation Analytics")

    try:
        # Fetch tasks for analytics
        response = requests.get(f"{API_URL}/tasks", params={"limit": 100})
        if response.status_code == 200:
            analytics_tasks = response.json()["tasks"]

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

            else:
                st.info("No analytics data available yet. Generate videos to see analytics.")
        else:
            st.error("Failed to fetch analytics data")
    except Exception as e:
        st.error(f"Error generating analytics: {e}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center">
        <p>AI Video Generator | Made with ❤️ using Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)