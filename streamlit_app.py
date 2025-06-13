import streamlit as st
import subprocess
import os
from pathlib import Path
import tempfile
from PIL import Image
import time

# Set page config
st.set_page_config(
    page_title="Qwen2-VL RKLLM Inference",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 Qwen2-VL RKLLM Inference App")
st.markdown("""
This app allows you to run Qwen2-VL model inference using RKLLM runtime.
Upload an image and configure the model parameters to get AI-generated responses.
""")

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
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            image_path = tmp_file.name
    elif use_demo and Path("data/demo.jpg").exists():
        # Use demo image
        image_path = "./data/demo.jpg"
        demo_image = Image.open(image_path)
        st.image(demo_image, caption="Demo Image", use_column_width=True)
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
        
        if st.button("🔥 Run Inference", type="primary", use_container_width=True):
            with st.spinner("Running inference..."):
                try:
                    # Run the command
                    result = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minute timeout
                    )
                    
                    if result.returncode == 0:
                        st.success("✅ Inference completed successfully!")
                        
                        # Display output
                        st.subheader("📝 Model Output")
                        if result.stdout:
                            st.text_area(
                                "Standard Output:",
                                value=result.stdout,
                                height=200,
                                disabled=True
                            )
                        
                        if result.stderr:
                            st.text_area(
                                "Standard Error:",
                                value=result.stderr,
                                height=100,
                                disabled=True
                            )
                    else:
                        st.error(f"❌ Inference failed with return code: {result.returncode}")
                        if result.stderr:
                            st.error(f"Error: {result.stderr}")
                        if result.stdout:
                            st.info(f"Output: {result.stdout}")
                            
                except subprocess.TimeoutExpired:
                    st.error("❌ Inference timed out (5 minutes)")
                except Exception as e:
                    st.error(f"❌ Error running inference: {str(e)}")
                    
                finally:
                    # Clean up temporary file if it was created
                    if uploaded_file is not None and os.path.exists(image_path):
                        try:
                            os.unlink(image_path)
                        except:
                            pass
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

### 🔧 Command Format:
```bash
./app/build/app <image_path> <vision_model.rknn> <llm_model.rkllm> <max_new_tokens> <max_context_len> <core_num>
```
""")