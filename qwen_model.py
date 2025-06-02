# qwen_model.py
import numpy as np
import logging
import os
from typing import Optional
from ctypes import *
import sys

from rknn_bindings import (
    rkllm_lib, RKLLMParam, RKLLMExtendParam, RKLLMInput, RKLLMMultimodalInput, 
    RKLLMInferParam, RKLLMResult, CallbackFunc,
    RKLLM_RUN_NORMAL, RKLLM_RUN_FINISH, RKLLM_RUN_ERROR,
    RKLLM_INFER_GENERATE, RKLLM_INPUT_PROMPT, RKLLM_INPUT_MULTIMODAL
)

logger = logging.getLogger(__name__)

# Define a proper callback function type
def callback_func(result, userdata, state):
    """Callback function to capture model output"""
    if state == RKLLM_RUN_NORMAL and result and result.contents.text:
        # Get the model instance from userdata
        model_instance = cast(userdata, py_object).value
        if hasattr(model_instance, 'current_response'):
            # Append new text to the current response
            new_text = result.contents.text.decode('utf-8', errors='ignore')
            model_instance.current_response += new_text
    return None

# Create the callback function pointer
CALLBACK_FUNC = CallbackFunc(callback_func)

class QwenVLModel:
    def __init__(self, model_path: str, max_new_tokens: int = 256, max_context_len: int = 1024):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.max_context_len = max_context_len
        self.llm_handle = None
        self.is_initialized = False
        self.chat_history = []
        self.current_response = ""
        
    def init_model(self) -> bool:
        try:
            # Validate model file
            if not os.path.exists(self.model_path):
                print(f"❌ Model file does not exist: {self.model_path}")
                return False
                
            print("Setting up parameters...")
            param = RKLLMParam()
            param.model_path = self.model_path.encode('utf-8')
            param.max_new_tokens = self.max_new_tokens
            param.max_context_len = self.max_context_len
            param.top_k = 1
            param.top_p = c_float(0.9)
            param.temperature = c_float(0.8)
            param.repeat_penalty = c_float(1.1)
            param.frequency_penalty = c_float(0.0)
            param.presence_penalty = c_float(0.0)
            param.mirostat = 0
            param.mirostat_tau = c_float(5.0)
            param.mirostat_eta = c_float(0.1)
            param.skip_special_token = True
            param.is_async = False
            
            # Image tokens - make sure these are properly null-terminated
            param.img_start = b"<|vision_start|>\0"
            param.img_end = b"<|vision_end|>\0"
            param.img_content = b"<|image_pad|>\0"
            
            # Extended parameters
            print("Setting up extended parameters...")
            extend_param = RKLLMExtendParam()
            extend_param.base_domain_id = 1
            param.extend_param = extend_param
            
            # Initialize with proper callback
            print("Calling rkllm_init...")
            self.llm_handle = c_void_p()
            
            # Use the global callback function
            ret = rkllm_lib.rkllm_init(byref(self.llm_handle), byref(param), CALLBACK_FUNC)
            
            if ret != 0:
                raise Exception(f"rkllm_init failed with code {ret}")
            
            print("rkllm_init successful")
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"❌ Model initialization failed: {e}")
            logger.error(f"Model initialization failed: {e}")
            return False

    def generate_response(self, prompt: str, image_features: Optional[np.ndarray] = None) -> str:
        if not self.is_initialized:
            return "Error: Model not initialized"
            
        try:
            print(f"Generating response for prompt: {prompt[:50]}...")
            self.current_response = ""
            
            rkllm_input = RKLLMInput()
            
            if image_features is not None and "<image>" in prompt:
                print("Using multimodal input...")
                # Multimodal input
                rkllm_input.input_type = RKLLM_INPUT_MULTIMODAL
                
                # Ensure image_features is the right type and size
                if len(image_features) > 196 * 1536:
                    image_features = image_features[:196 * 1536]
                
                image_embed_array = (c_float * len(image_features))(*image_features.astype(np.float32))
                
                rkllm_input.multimodal_input.prompt = prompt.encode('utf-8')
                rkllm_input.multimodal_input.image_embed = image_embed_array
                rkllm_input.multimodal_input.n_image_tokens = min(196, len(image_features) // 1536)
                rkllm_input.multimodal_input.n_image = 1
                rkllm_input.multimodal_input.image_height = 392
                rkllm_input.multimodal_input.image_width = 392
            else:
                print("Using text-only input...")
                # Text-only input
                rkllm_input.input_type = RKLLM_INPUT_PROMPT
                rkllm_input.prompt_input = prompt.encode('utf-8')
            
            # Set inference parameters
            infer_params = RKLLMInferParam()
            infer_params.mode = RKLLM_INFER_GENERATE
            infer_params.keep_history = 0
            
            # Create a pointer to self for the callback
            self_ptr = py_object(self)
            
            print("Running inference...")
            ret = rkllm_lib.rkllm_run(self.llm_handle, byref(rkllm_input), byref(infer_params), byref(self_ptr))
            
            if ret == 0:
                print("✅ Response generated successfully")
                return self.current_response if self.current_response else "Response generated but empty"
            else:
                raise Exception(f"rkllm_run failed with code {ret}")
                
        except Exception as e:
            print(f"❌ Response generation failed: {e}")
            logger.error(f"Response generation failed: {e}")
            return f"Error: {str(e)}"
            
    def clear_history(self):
        """Clear the KV cache"""
        if self.is_initialized and self.llm_handle:
            try:
                ret = rkllm_lib.rkllm_clear_kv_cache(self.llm_handle, 0)
                if ret == 0:
                    print("✅ KV cache cleared")
                else:
                    print(f"⚠️ Failed to clear KV cache: {ret}")
            except Exception as e:
                print(f"❌ Error clearing KV cache: {e}")
                
    def release(self):
        """Release model resources"""
        if self.is_initialized and self.llm_handle:
            try:
                print("Releasing RKLLM handle...")
                ret = rkllm_lib.rkllm_destroy(self.llm_handle)
                if ret == 0:
                    print("✅ RKLLM handle released")
                else:
                    print(f"⚠️ RKLLM destroy returned: {ret}")
                    
                self.llm_handle = None
                self.is_initialized = False
                
            except Exception as e:
                print(f"❌ Error releasing model: {e}")
                logger.error(f"Error releasing model: {e}")