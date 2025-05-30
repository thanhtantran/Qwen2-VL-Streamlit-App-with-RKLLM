# main.py
import streamlit as st
import numpy as np
from PIL import Image
import time
import logging
import atexit
import traceback
import sys
import os

# Minimal logging - only errors
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Model paths
ENCODER_MODEL_PATH = "./models/Qwen2-VL-2B-RKLLM/qwen2_vl_2b_vision_rk3588.rknn"
LLM_MODEL_PATH = "./models/Qwen2-VL-2B-RKLLM/qwen2-vl-llm_rk3588.rkllm"

def cleanup_models():
    """Cleanup function to release model resources"""
    try:
        if 'image_encoder' in st.session_state and st.session_state.image_encoder:
            st.session_state.image_encoder.release()
        if 'qwen_model' in st.session_state and st.session_state.qwen_model:
            st.session_state.qwen_model.release()
    except:
        pass

atexit.register(cleanup_models)

def safe_import_modules():
    """Import modules with error handling"""
    try:
        from image_encoder import ImageEncoder
        from qwen_model import QwenVLModel
        return ImageEncoder, QwenVLModel, None
    except Exception as e:
        return None, None, str(e)

def main():
    st.set_page_config(
        page_title="Qwen2-VL Image Q&A",
        page_icon="🖼️",
        layout="wide"
    )
    
    st.title("🖼️ Qwen2-VL Image Question & Answer")
    
    # Import modules
    ImageEncoder, QwenVLModel, import_error = safe_import_modules()
    if import_error:
        st.error(f"❌ Module Import Failed: {import_error}")
        st.stop()
    
    # Sidebar for model configuration
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        
        # Model parameters
        max_new_tokens = st.slider("Max New Tokens", 50, 1000, 512)
        max_context_len = st.slider("Max Context Length", 512, 4096, 2048)
        core_num = st.selectbox("NPU Core Number", [1, 2, 3], index=0)
        
        # Initialize models button
        if st.button("🚀 Initialize Models"):
            
            # Create containers for status updates
            status_container = st.container()
            error_container = st.container()
            
            with status_container:
                st.info("🔄 Starting model initialization...")
            
            try:
                # Step 1: Cleanup
                cleanup_models()
                
                # Step 2: Initialize Image Encoder
                with status_container:
                    st.info("🔄 Creating Image Encoder...")
                
                st.session_state.image_encoder = ImageEncoder(ENCODER_MODEL_PATH, core_num)
                
                with status_container:
                    st.info("🔄 Initializing RKNN model...")
                
                encoder_success = st.session_state.image_encoder.init_encoder()
                
                if not encoder_success:
                    with error_container:
                        st.error("❌ Image encoder initialization failed")
                    st.stop()
                
                with status_container:
                    st.success("✅ Image encoder initialized")
                
                # Step 3: Initialize LLM Model
                with status_container:
                    st.info("🔄 Creating LLM Model...")
                
                st.session_state.qwen_model = QwenVLModel(LLM_MODEL_PATH, max_new_tokens, max_context_len)
                
                with status_container:
                    st.info("🔄 Initializing RKLLM model...")
                
                model_success = st.session_state.qwen_model.init_model()
                
                if not model_success:
                    with error_container:
                        st.error("❌ LLM model initialization failed")
                    st.stop()
                
                with status_container:
                    st.success("✅ LLM model initialized")
                
                # Success
                if encoder_success and model_success:
                    st.session_state.models_initialized = True
                    with status_container:
                        st.success("🎉 All models initialized successfully!")
                
            except Exception as e:
                with error_container:
                    st.error("❌ **Initialization Failed:**")
                    st.code(f"Error: {str(e)}")
                    
                    # Show more details in expander
                    with st.expander("🔍 Full Error Details"):
                        st.code(traceback.format_exc())
                
                st.session_state.models_initialized = False
        
        # Model status
        if st.session_state.get('models_initialized', False):
            st.success("🟢 Models Ready")
        else:
            st.warning("🔴 Models Not Initialized")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff']
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                st.session_state.current_image = image
                
                if st.session_state.get('models_initialized', False):
                    if st.button("🔄 Encode Image"):
                        try:
                            with st.spinner("Encoding image..."):
                                image_features = st.session_state.image_encoder.encode_image(image)
                                
                                if image_features is not None:
                                    st.session_state.image_features = image_features
                                    st.success(f"✅ Image encoded! Shape: {image_features.shape}")
                                else:
                                    st.error("❌ Failed to encode image")
                        except Exception as e:
                            st.error(f"❌ Encoding error: {str(e)}")
                else:
                    st.warning("⚠️ Initialize models first")
                    
            except Exception as e:
                st.error(f"❌ Image processing error: {str(e)}")
    
    with col2:
        st.header("💬 Ask Questions")
        
        # Quick questions
        quick_questions = [
            "<image>What is in the image?",
            "<image>Describe the main objects in this image",
            "<image>What colors are prominent in this image?",
            "<image>What is the setting or location of this image?"
        ]
        
        for i, question in enumerate(quick_questions):
            if st.button(f"❓ {question.replace('<image>', '')}", key=f"quick_{i}"):
                st.session_state.current_question = question
        
        # Custom question
        custom_question = st.text_area(
            "Custom question:",
            placeholder="<image>What do you see in this image?"
        )
        
        if st.button("🔍 Ask Question"):
            if custom_question:
                st.session_state.current_question = custom_question
            else:
                st.warning("Please enter a question")
        
        # Generate response
        if st.session_state.get('current_question') and st.session_state.get('models_initialized', False):
            question = st.session_state.current_question
            
            st.subheader("🤖 Response")
            
            try:
                with st.spinner("Generating response..."):
                    if st.session_state.get('current_image') is not None and "<image>" in question:
                        if 'image_features' in st.session_state:
                            image_features = st.session_state.image_features
                            response = st.session_state.qwen_model.generate_response(question, image_features)
                        else:
                            st.warning("⚠️ Please encode the image first")
                            response = None
                    else:
                        response = st.session_state.qwen_model.generate_response(question)
                    
                    if response:
                        st.write("**Question:**", question.replace("<image>", "🖼️"))
                        st.write("**Answer:**", response)
                        
            except Exception as e:
                st.error(f"❌ Response error: {str(e)}")
        
        elif st.session_state.get('current_question'):
            st.warning("⚠️ Please initialize models first")

if __name__ == "__main__":
    if 'models_initialized' not in st.session_state:
        st.session_state.models_initialized = False
    
    try:
        main()
    except Exception as e:
        st.error(f"❌ Critical error: {str(e)}")