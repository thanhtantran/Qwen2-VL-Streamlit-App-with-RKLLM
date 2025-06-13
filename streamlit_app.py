import streamlit as st
import subprocess
import os
from pathlib import Path
from PIL import Image
import time
from datetime import datetime
import threading
import queue

# Set page config
st.set_page_config(
    page_title="Qwen2-VL RKLLM Inference",
    page_icon="🤖",
    layout="wide"
)

# Header with logo
col_title, col_logo = st.columns([3, 1])
with col_title:
    st.title("🤖 Qwen2-VL RKLLM Inference App")
    st.markdown("""
    This app allows you to run Qwen2-VL model inference using RKLLM runtime.
    Upload an image and configure the model parameters to get AI-generated responses.
    """)
with col_logo:
    # Orange Pi Vietnam logo
    st.markdown("""
    <div style="text-align: right; padding-top: 20px;">
        <a href="https://orangepi.vn" target="_blank">
            <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIwIiBoZWlnaHQ9IjgwIiB2aWV3Qm94PSIwIDAgMTIwIDgwIiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8Y2lyY2xlIGN4PSIyNSIgY3k9IjI1IiByPSIyMCIgZmlsbD0iI0ZGQTUwMCIvPgo8cGF0aCBkPSJNMTUgMjVMMzUgMjUiIHN0cm9rZT0iI0ZGRkZGRiIgc3Ryb2tlLXdpZHRoPSIyIi8+CjxwYXRoIGQ9Ik0yNSAxNUwyNSAzNSIgc3Ryb2tlPSIjRkZGRkZGIiBzdHJva2Utd2lkdGg9IjIiLz4KPHN0YXIgY3g9IjkwIiBjeT0iNDAiIHI9IjEwIiBmaWxsPSIjRkYwMDAwIi8+Cjx0ZXh0IHg9IjUwIiB5PSIzNSIgZm9udC1mYW1pbHk9IkFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjE0IiBmaWxsPSIjMzMzMzMzIj5PcmFuZ2UgUGk8L3RleHQ+Cjx0ZXh0IHg9IjUwIiB5PSI1NSIgZm9udC1mYW1pbHk9IkFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyIiBmaWxsPSIjNjY2NjY2Ij5WaWV0bmFtPC90ZXh0Pgo8L3N2Zz4K" alt="Orange Pi Vietnam" style="height: 60px;">
        </a>
    </div>
    """, unsafe_allow_html=True)

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# Model selection
st.sidebar.subheader("Model Selection")
models_dir = Path("models")
if models_dir.exists():
    model_folders = [f for f in models_dir.iterdir() if f.is_dir()]
    if model_folders:
        selected_model = st.sidebar.selectbox(
            "Choose Model:",
            options=[f.name for f in model_folders],
            help="Select the model variant to use for inference"
        )
        
        # Get model paths
        model_path = models_dir / selected_model
        vision_model = None
        llm_model = None
        
        # Find .rknn and .rkllm files
        for file in model_path.iterdir():
            if file.suffix == '.rknn':
                vision_model = str(file)
            elif file.suffix == '.rkllm':
                llm_model = str(file)
        
        if vision_model and llm_model:
            st.sidebar.success(f"✅ Model found: {selected_model}")
            st.sidebar.text(f"Vision: {Path(vision_model).name}")
            st.sidebar.text(f"LLM: {Path(llm_model).name}")
        else:
            st.sidebar.error("❌ Model files not found")
    else:
        st.sidebar.error("❌ No model folders found")
else:
    st.sidebar.error("❌ Models directory not found")

# Parameters configuration
st.sidebar.subheader("Model Parameters")

max_new_tokens = st.sidebar.slider(
    "Max New Tokens",
    min_value=1,
    max_value=2048,
    value=128,
    step=1,
    help="Maximum number of new tokens to generate (argument 4)"
)

max_context_len = st.sidebar.slider(
    "Max Context Length",
    min_value=1,
    max_value=4096,
    value=512,
    step=1,
    help="Maximum context length for the model (argument 5)"
)

core_num = st.sidebar.selectbox(
    "NPU Core Configuration",
    options=[1, 2, 3],
    index=2,  # Default to 3
    help="Number of NPU cores to use:\n- 1: Single core (AUTO)\n- 2: Dual core (0+1)\n- 3: Triple core (0+1+2)"
)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Image Input")
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image for the model to analyze"
    )
    
    # Option to use demo image
    use_demo = st.checkbox("Use demo image (./data/demo.jpg)", value=True)
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Save uploaded file to data folder with datetime name
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = uploaded_file.name.split('.')[-1]
        filename = f"{timestamp}.{file_extension}"
        image_path = data_dir / filename
        
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        image_path = str(image_path)
    elif use_demo and Path("data/demo.jpg").exists():
        # Use demo image
        image_path = "./data/demo.jpg"
        demo_image = Image.open(image_path)
        st.image(demo_image, caption="Demo Image", use_container_width=True)
    else:
        image_path = None
        st.info("Please upload an image or use the demo image.")

with col2:
    st.subheader("🚀 Inference")
    
    # Check if all requirements are met
    can_run = (
        image_path is not None and
        'vision_model' in locals() and vision_model is not None and
        'llm_model' in locals() and llm_model is not None and
        Path("app/build/app").exists()
    )
    
    if can_run:
        st.success("✅ Ready to run inference")
        
        # Display command that will be executed
        command = [
            "./app/build/app",
            image_path,
            vision_model,
            llm_model,
            str(max_new_tokens),
            str(max_context_len),
            str(core_num)
        ]
        
        st.code(" ".join(command), language="bash")
        
        # Initialize session state for chat
        if 'chat_process' not in st.session_state:
            st.session_state.chat_process = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'process_output' not in st.session_state:
            st.session_state.process_output = ""
        if 'model_ready' not in st.session_state:
            st.session_state.model_ready = False
            
        # Start inference button
        if st.button("🔥 Start Interactive Chat", type="primary", use_container_width=True):
            if st.session_state.chat_process is None:
                try:
                    # Start the process
                    st.session_state.chat_process = subprocess.Popen(
                        command,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1,
                        universal_newlines=True
                    )
                    st.session_state.model_ready = False
                    st.session_state.process_output = ""
                    st.session_state.chat_history = []
                    st.success("✅ Starting inference process...")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error starting process: {str(e)}")
        
        # Stop inference button
        if st.session_state.chat_process is not None:
            if st.button("🛑 Stop Chat", type="secondary", use_container_width=True):
                try:
                    st.session_state.chat_process.terminate()
                    st.session_state.chat_process.wait(timeout=5)
                except:
                    st.session_state.chat_process.kill()
                finally:
                    st.session_state.chat_process = None
                    st.session_state.model_ready = False
                    st.session_state.process_output = ""
                    st.success("✅ Chat stopped")
                    st.rerun()
        
        # Monitor process output
        if st.session_state.chat_process is not None:
            try:
                # Check if process is still running
                if st.session_state.chat_process.poll() is not None:
                    st.error("❌ Process has terminated")
                    st.session_state.chat_process = None
                    st.session_state.model_ready = False
                    st.rerun()
                else:
                    # Read available output
                    import select
                    import sys
                    
                    # For Windows compatibility, use a different approach
                    if hasattr(select, 'select'):
                        ready, _, _ = select.select([st.session_state.chat_process.stdout], [], [], 0.1)
                        if ready:
                            line = st.session_state.chat_process.stdout.readline()
                            if line:
                                st.session_state.process_output += line
                                if "user:" in line and not st.session_state.model_ready:
                                    st.session_state.model_ready = True
                    else:
                        # Windows fallback - check periodically
                        time.sleep(0.1)
                        
                    # Display current output
                    if st.session_state.process_output:
                        st.text_area(
                            "📟 Process Output:",
                            value=st.session_state.process_output,
                            height=300,
                            disabled=True
                        )
                    
                    # Show chat interface when model is ready
                    if st.session_state.model_ready:
                        st.subheader("💬 Chat Interface")
                        
                        # Quick question buttons
                        col_q1, col_q2 = st.columns(2)
                        with col_q1:
                            if st.button("❓ What is in the image?"):
                                try:
                                    st.session_state.chat_process.stdin.write("0\n")
                                    st.session_state.chat_process.stdin.flush()
                                    st.session_state.chat_history.append("User: [0] What is in the image?")
                                except Exception as e:
                                    st.error(f"Error sending input: {e}")
                        
                        with col_q2:
                            if st.button("🌐 Trong bức ảnh có gì?"):
                                try:
                                    st.session_state.chat_process.stdin.write("1\n")
                                    st.session_state.chat_process.stdin.flush()
                                    st.session_state.chat_history.append("User: [1] Trong bức ảnh có gì?")
                                except Exception as e:
                                    st.error(f"Error sending input: {e}")
                        
                        # Custom input
                        user_input = st.text_input(
                            "💭 Or enter your custom question:",
                            placeholder="Type your question here...",
                            key="user_question"
                        )
                        
                        if st.button("📤 Send Custom Question") and user_input:
                            try:
                                st.session_state.chat_process.stdin.write(f"{user_input}\n")
                                st.session_state.chat_process.stdin.flush()
                                st.session_state.chat_history.append(f"User: {user_input}")
                                st.session_state.user_question = ""  # Clear input
                            except Exception as e:
                                st.error(f"Error sending input: {e}")
                        
                        # Display chat history
                        if st.session_state.chat_history:
                            st.subheader("📜 Chat History")
                            for message in st.session_state.chat_history:
                                st.text(message)
                    
                    # Auto-refresh every 2 seconds when process is running
                    time.sleep(2)
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error monitoring process: {str(e)}")
                st.session_state.chat_process = None
                st.session_state.model_ready = False
    else:
        st.warning("⚠️ Requirements not met:")
        if image_path is None:
            st.write("- No image selected")
        if 'vision_model' not in locals() or vision_model is None:
            st.write("- Vision model not found")
        if 'llm_model' not in locals() or llm_model is None:
            st.write("- LLM model not found")
        if not Path("app/build/app").exists():
            st.write("- App executable not found (./app/build/app)")

# Footer with information
st.markdown("---")
st.markdown("""
### 📋 Parameter Information:
- **Max New Tokens**: Maximum number of tokens the model will generate
- **Max Context Length**: Maximum length of the input context the model can process
- **NPU Core Configuration**: Number of NPU cores to utilize for inference
  - 1: Single core (AUTO mode)
  - 2: Dual core (cores 0+1)
  - 3: Triple core (cores 0+1+2)
""")

# Copyright footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #666666;">
    © 2025 Copyright by <a href="https://orangepi.vn" target="_blank" style="color: #FF6B35; text-decoration: none;">Orange Pi Vietnam</a>
</div>
""", unsafe_allow_html=True)