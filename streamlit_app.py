import streamlit as st
import subprocess
import os
from pathlib import Path
from PIL import Image
import time
from datetime import datetime
import threading
import queue
import select
import sys

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
    st.image("https://orangepi.vn/wp-content/uploads/2018/05/logo1-1.png", width=120)

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
            st.sidebar.success(f"✅ Model loaded: {selected_model}")
            st.sidebar.write(f"Vision: {Path(vision_model).name}")
            st.sidebar.write(f"LLM: {Path(llm_model).name}")
        else:
            st.sidebar.error("❌ Model files not found")
            st.sidebar.write("Required: .rknn and .rkllm files")
    else:
        st.sidebar.warning("⚠️ No model folders found in 'models' directory")
else:
    st.sidebar.error("❌ 'models' directory not found")

# Parameters configuration
st.sidebar.subheader("Parameters")
max_new_tokens = st.sidebar.slider(
    "Max New Tokens",
    min_value=1,
    max_value=2048,
    value=512,
    help="Maximum number of tokens to generate"
)

max_context_length = st.sidebar.slider(
    "Max Context Length",
    min_value=512,
    max_value=8192,
    value=4096,
    help="Maximum context length for the model"
)

npu_core_num = st.sidebar.selectbox(
    "NPU Core Configuration",
    options=[1, 2, 3],
    index=0,
    help="Number of NPU cores to use (1=AUTO, 2=cores 0+1, 3=cores 0+1+2)"
)

# Main content area
st.header("📸 Image Input")

# Image upload
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=['png', 'jpg', 'jpeg'],
    help="Upload an image for the model to analyze"
)

image_path = None
if uploaded_file is not None:
    # Save uploaded file
    image_path = f"temp_{uploaded_file.name}"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display image
    image = Image.open(image_path)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.success(f"✅ Image uploaded: {uploaded_file.name}")
else:
    # Use demo image if available
    demo_path = Path("data/demo.jpg")
    if demo_path.exists():
        image_path = str(demo_path)
        image = Image.open(image_path)
        st.image(image, caption="Demo Image (data/demo.jpg)", use_column_width=True)
        st.info("ℹ️ Using demo image. Upload your own image above to replace it.")
    else:
        st.warning("⚠️ No image uploaded and demo image not found")

# Inference section
st.header("🚀 Inference")

# Check if all requirements are met
if (image_path is not None and 
    'vision_model' in locals() and vision_model is not None and 
    'llm_model' in locals() and llm_model is not None and 
    Path("app/build/app").exists()):
    
    # Build command
    command = [
        "./app/build/app",
        vision_model,
        llm_model,
        image_path,
        str(max_new_tokens),
        str(max_context_length),
        str(npu_core_num)
    ]
    
    st.write(f"**Command:** `{' '.join(command)}`")
    
    # Initialize session state
    if 'chat_process' not in st.session_state:
        st.session_state.chat_process = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'process_output' not in st.session_state:
        st.session_state.process_output = ""
    if 'model_ready' not in st.session_state:
        st.session_state.model_ready = False
    if 'output_buffer' not in st.session_state:
        st.session_state.output_buffer = ""
        
    # Start inference button
    if st.button("🔥 Start Interactive Chat", type="primary", use_container_width=True):
        if st.session_state.chat_process is None:
            try:
                # Start the process with optimized buffering
                st.session_state.chat_process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # Merge stderr with stdout
                    text=True,
                    bufsize=0,  # Unbuffered
                    universal_newlines=True,
                    env=dict(os.environ, PYTHONUNBUFFERED="1")  # Force unbuffered output
                )
                st.session_state.model_ready = False
                st.session_state.process_output = ""
                st.session_state.output_buffer = ""
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
                st.session_state.output_buffer = ""
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
                # Read available output with improved buffering
                try:
                    # Use select for non-blocking read (Unix-like systems)
                    if hasattr(select, 'select'):
                        ready, _, _ = select.select([st.session_state.chat_process.stdout], [], [], 0.1)
                        if ready:
                            # Read character by character to avoid buffering issues
                            char = st.session_state.chat_process.stdout.read(1)
                            if char:
                                st.session_state.output_buffer += char
                                st.session_state.process_output += char
                                
                                # Check for complete lines or specific patterns
                                if char == '\n' or st.session_state.output_buffer.endswith('user:'):
                                    # Check if model is ready - look for "user:" at the end
                                    if (st.session_state.process_output.strip().endswith("user:") and 
                                        not st.session_state.model_ready):
                                        st.session_state.model_ready = True
                                        st.success("🟢 Model is ready for input!")
                                    
                                    # Clear buffer after processing
                                    if char == '\n':
                                        st.session_state.output_buffer = ""
                    else:
                        # Fallback for systems without select (like Windows)
                        # Use a small timeout and read available data
                        import fcntl
                        import errno
                        
                        try:
                            # Set non-blocking mode
                            fd = st.session_state.chat_process.stdout.fileno()
                            fl = fcntl.fcntl(fd, fcntl.F_GETFL)
                            fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
                            
                            # Try to read
                            data = st.session_state.chat_process.stdout.read()
                            if data:
                                st.session_state.process_output += data
                                # Check if model is ready
                                if (st.session_state.process_output.strip().endswith("user:") and 
                                    not st.session_state.model_ready):
                                    st.session_state.model_ready = True
                                    st.success("🟢 Model is ready for input!")
                        except (IOError, OSError) as e:
                            if e.errno != errno.EAGAIN:
                                raise
                        except ImportError:
                            # fcntl not available, use basic approach
                            pass
                            
                except Exception as e:
                    # Continue on read errors
                    pass
                    
                # Display current output
                if st.session_state.process_output:
                    # Show last 2000 characters to avoid UI slowdown
                    display_output = st.session_state.process_output[-2000:] if len(st.session_state.process_output) > 2000 else st.session_state.process_output
                    st.text_area(
                        "📟 Process Output:",
                        value=display_output,
                        height=300,
                        disabled=True
                    )
                
                # Show loading status
                if not st.session_state.model_ready:
                    st.info("🔄 Model is loading... Please wait for the 'user:' prompt.")
                
                # Show chat interface when model is ready
                if st.session_state.model_ready:
                    st.subheader("💬 Chat Interface")
                    st.success("🟢 Model is ready for questions!")
                    
                    # Quick question buttons
                    col_q1, col_q2 = st.columns(2)
                    with col_q1:
                        if st.button("❓ What is in the image?"):
                            try:
                                st.session_state.chat_process.stdin.write("0\n")
                                st.session_state.chat_process.stdin.flush()
                                st.session_state.chat_history.append("User: [0] What is in the image?")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error sending input: {e}")
                    
                    with col_q2:
                        if st.button("🌐 Trong bức ảnh có gì?"):
                            try:
                                st.session_state.chat_process.stdin.write("1\n")
                                st.session_state.chat_process.stdin.flush()
                                st.session_state.chat_history.append("User: [1] Trong bức ảnh có gì?")
                                st.rerun()
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
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error sending input: {e}")
                    
                    # Display chat history
                    if st.session_state.chat_history:
                        st.subheader("📜 Chat History")
                        for message in st.session_state.chat_history:
                            if message.startswith("User:"):
                                st.markdown(f"**{message}**")
                            else:
                                st.markdown(message)
                
                # Auto-refresh every 1 second when process is running
                time.sleep(1)
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
st.markdown("© 2025 Copyright by [Orange Pi Vietnam](https://orangepi.vn)")