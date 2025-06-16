import streamlit as st
import subprocess
import fcntl
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

class ProcessManager:
    def __init__(self):
        self.process = None
        self.output_queue = queue.Queue()
        self.input_queue = queue.Queue()
        self.output_thread = None
        self.input_thread = None
        self.running = False
        
    def start_process(self, command):
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            self.running = True
            
            # Start output reading thread
            self.output_thread = threading.Thread(target=self._read_output, daemon=True)
            self.output_thread.start()
            
            # Start input writing thread
            self.input_thread = threading.Thread(target=self._write_input, daemon=True)
            self.input_thread.start()
            
            return True
        except Exception as e:
            st.error(f"Failed to start process: {e}")
            return False
    
    def _read_output(self):
        """Read process output in separate thread"""
        try:
            while self.running and self.process and self.process.poll() is None:
                line = self.process.stdout.readline()
                if line:
                    self.output_queue.put(line)
                else:
                    time.sleep(0.1)
        except Exception as e:
            self.output_queue.put(f"Error reading output: {e}\n")
        finally:
            self.running = False
    
    def _write_input(self):
        """Write input to process in separate thread"""
        try:
            while self.running and self.process and self.process.poll() is None:
                try:
                    command = self.input_queue.get(timeout=0.1)
                    if command:
                        self.process.stdin.write(command + "\n")
                        self.process.stdin.flush()
                except queue.Empty:
                    continue
                except Exception as e:
                    st.error(f"Error writing input: {e}")
                    break
        except Exception as e:
            st.error(f"Input thread error: {e}")
    
    def send_command(self, command):
        """Send command to process"""
        if self.running:
            self.input_queue.put(command)
    
    def get_output(self):
        """Get all available output"""
        output_lines = []
        try:
            while True:
                line = self.output_queue.get_nowait()
                output_lines.append(line)
        except queue.Empty:
            pass
        return output_lines
    
    def is_running(self):
        """Check if process is still running"""
        return self.running and self.process and self.process.poll() is None
    
    def stop(self):
        """Stop the process"""
        self.running = False
        if self.process:
            self.process.terminate()
            self.process = None

# Initialize process manager in session state
if 'process_manager' not in st.session_state:
    st.session_state.process_manager = ProcessManager()
if 'process_output' not in st.session_state:
    st.session_state.process_output = ""
if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Header with logo
col_title, col_logo = st.columns([3, 1])
with col_title:
    st.title("🤖 Qwen2-VL RKLLM Inference App")
    st.markdown("""
    This app allows you to run Qwen2-VL model inference using RKLLM runtime.
    Upload an image and configure the model parameters to get AI-generated responses.
    """)
with col_logo:
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
    index=2,
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
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.success(f"✅ Image uploaded: {uploaded_file.name}")
else:
    # Use demo image if available
    demo_path = Path("data/demo.jpg")
    if demo_path.exists():
        image_path = str(demo_path)
        image = Image.open(image_path)
        st.image(image, caption="Demo Image (data/demo.jpg)", use_container_width=True)
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
        image_path,        
        vision_model,
        llm_model,
        str(max_new_tokens),
        str(max_context_length),
        str(npu_core_num)
    ]
    
    st.write(f"**Command:** `{' '.join(command)}`")
    
    # Control buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔥 Start Interactive Chat", type="primary", use_container_width=True):
            if not st.session_state.process_manager.is_running():
                if st.session_state.process_manager.start_process(command):
                    st.session_state.process_output = ""
                    st.session_state.model_ready = False
                    st.session_state.chat_history = []
                    st.success("🚀 Process started!")
                    time.sleep(0.5)  # Give process time to start
                    st.rerun()
            else:
                st.warning("Process is already running!")
    
    with col2:
        if st.button("🛑 Stop Process", type="secondary", use_container_width=True):
            if st.session_state.process_manager.is_running():
                st.session_state.process_manager.stop()
                st.session_state.model_ready = False
                st.success("Process stopped!")
                st.rerun()
    
    # Monitor process output
    if st.session_state.process_manager.is_running():
        # Get new output
        new_output = st.session_state.process_manager.get_output()
        if new_output:
            for line in new_output:
                st.session_state.process_output += line
                
                # Check if model is ready
                if (st.session_state.process_output.strip().endswith("*************************************************************************") and 
                    not st.session_state.model_ready):
                    st.session_state.model_ready = True
        
        # Display process output
        if st.session_state.process_output:
            display_output = st.session_state.process_output[-2000:] if len(st.session_state.process_output) > 2000 else st.session_state.process_output
            st.text_area(
                "📟 Process Output:",
                value=display_output,
                height=300,
                disabled=True
            )
        
        # Show status
        if not st.session_state.model_ready:
            st.info("🔄 Model is loading... Please wait for the user prompt.")
        else:
            st.success("🟢 Model is ready for questions!")
            
            # Chat interface
            st.subheader("💬 Chat Interface")
            
            # Quick question buttons
            col_q1, col_q2 = st.columns(2)
            with col_q1:
                if st.button("❓ What is in the image?"):
                    st.session_state.process_manager.send_command("0")
                    st.session_state.chat_history.append("User: [0] What is in the image?")
                    st.session_state.waiting_for_response = True
                    st.rerun()
            
            with col_q2:
                if st.button("🌐 Trong bức ảnh có gì?"):
                    st.session_state.process_manager.send_command("1")
                    st.session_state.chat_history.append("User: [1] Trong bức ảnh có gì?")
                    st.session_state.waiting_for_response = True
                    st.rerun()
            
            # Custom input
            user_input = st.text_input(
                "💭 Or enter your custom question:",
                placeholder="Type your question here...",
                key="user_question"
            )
            
            if st.button("📤 Send Custom Question") and user_input:
                # Add <image> prefix to custom questions
                formatted_question = f"<image>{user_input}"
                st.session_state.process_manager.send_command(formatted_question)
                st.session_state.chat_history.append(f"User: {user_input}")
                st.session_state.waiting_for_response = True
                st.rerun()
            
            # Display chat history
            if st.session_state.chat_history:
                st.subheader("📜 Chat History")
                for message in st.session_state.chat_history:
                    if message.startswith("User:"):
                        st.markdown(f"**{message}**")
                    elif message.startswith("Robot:"):
                        st.markdown(f"*{message}*")
                    else:
                        st.markdown(message)
        
        # Auto-refresh to get new output
        time.sleep(1)
        st.rerun()
    
    elif st.session_state.process_output:
        # Show last output even if process stopped
        st.text_area(
            "📟 Process Output (Stopped):",
            value=st.session_state.process_output[-2000:],
            height=300,
            disabled=True
        )

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

# Footer
st.markdown("---")
st.markdown("© 2025 Copyright by [Orange Pi Vietnam](https://orangepi.vn)")