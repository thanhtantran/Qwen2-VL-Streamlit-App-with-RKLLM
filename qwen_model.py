# qwen_model.py
import numpy as np
import logging
from typing import Optional
from ctypes import *

from rknn_bindings import (
    rkllm_lib, RKLLMParam, RKLLMExtendParam, RKLLMInput, RKLLMMultimodalInput, 
    RKLLMInferParam, RKLLMResult, CallbackFunc,
    RKLLM_RUN_NORMAL, RKLLM_RUN_FINISH, RKLLM_RUN_ERROR,
    RKLLM_INFER_GENERATE, RKLLM_INPUT_PROMPT, RKLLM_INPUT_MULTIMODAL
)

logger = logging.getLogger(__name__)

# Define a proper callback function type
def dummy_callback(result, userdata, state):
    """Dummy callback function"""
    pass

# Create the callback function pointer
CALLBACK_FUNC = CallbackFunc(dummy_callback)

class QwenVLModel:
    def __init__(self, model_path: str, max_new_tokens: int = 256, max_context_len: int = 1024):
        self.model_path = model_path.encode('utf-8')
        self.max_new_tokens = max_new_tokens
        self.max_context_len = max_context_len
        self.llm_handle = None
        self.is_initialized = False
        self.chat_history = []
        self.current_response = ""
        
    def init_model(self) -> bool:
        try:
            print("Setting up parameters...")
            param = RKLLMParam()
            param.model_path = self.model_path
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
            
            # Set chat template
            print("Setting chat template...")
            system_prompt = b"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\0"
            user_prompt = b"<|im_start|>user\n\0"
            assistant_prompt = b"<|im_end|>\n<|im_start|>assistant\n\0"
            
            ret = rkllm_lib.rkllm_set_chat_template(self.llm_handle, system_prompt, user_prompt, assistant_prompt)
            if ret != 0:
                print(f"Warning: rkllm_set_chat_template failed with code {ret}")
                # Continue anyway, might not be critical
            
            self.is_initialized = True
            print("✅ LLM model initialized successfully")
            return True
                
        except Exception as e:
            print(f"❌ QwenVLModel init failed: {e}")
            logger.error(f"QwenVLModel init failed: {e}")
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
            
            print("Running inference...")
            ret = rkllm_lib.rkllm_run(self.llm_handle, byref(rkllm_input), byref(infer_params), None)
            
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
        self.chat_history = []
        if self.is_initialized and self.llm_handle:
            try:
                rkllm_lib.rkllm_clear_kv_cache(self.llm_handle, 1)
            except:
                pass
        
    def release(self):
        if self.is_initialized and self.llm_handle:
            print("Releasing RKLLM handle...")
            try:
                rkllm_lib.rkllm_destroy(self.llm_handle)
                self.llm_handle = None
                self.is_initialized = False
                print("✅ RKLLM handle released")
            except Exception as e:
                print(f"Warning: Error releasing RKLLM handle: {e}")