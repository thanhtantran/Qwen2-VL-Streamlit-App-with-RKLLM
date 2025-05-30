# qwen_vl_model.py
import os
import signal
import logging
import ctypes
from ctypes import *
from typing import List, Optional, Dict, Any
import numpy as np

from rkllm_bindings import (
    rkllm_lib, RKLLMParam, RKLLMExtendParam, RKLLMLoraParam, RKLLMResult,
    RKLLMInferenceType, RKLLMTokenizerType, RKLLM_RUN_NORMAL, RKLLM_RUN_ASYNC
)

logger = logging.getLogger(__name__)

class QwenVLModel:
    def __init__(self, model_path: str, max_context_len: int = 512, max_new_tokens: int = 512):
        self.model_path = model_path
        self.max_context_len = max_context_len
        self.max_new_tokens = max_new_tokens
        self.handle = None
        self.is_initialized = False
        
        # Set up signal handler for debugging segfaults
        signal.signal(signal.SIGSEGV, self._segfault_handler)
        
    def _segfault_handler(self, signum, frame):
        print(f"❌ SEGMENTATION FAULT detected in QwenVL model!")
        print(f"Signal: {signum}")
        print(f"Frame: {frame}")
        import traceback
        traceback.print_stack(frame)
        exit(1)
        
    def _validate_model_file(self) -> bool:
        """Validate model file exists and is readable"""
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ Model file does not exist: {self.model_path}")
                return False
                
            if not os.path.isfile(self.model_path):
                print(f"❌ Model path is not a file: {self.model_path}")
                return False
                
            # Check file size
            file_size = os.path.getsize(self.model_path)
            if file_size < 1024:  # Less than 1KB is suspicious
                print(f"❌ Model file too small: {file_size} bytes")
                return False
                
            print(f"✅ Model file validated: {file_size} bytes")
            return True
            
        except Exception as e:
            print(f"❌ Error validating model file: {e}")
            return False
            
    def _init_param_structure(self) -> RKLLMParam:
        """Safely initialize parameter structure"""
        print("Initializing RKLLMParam structure...")
        
        # Create and zero-initialize the structure
        param = RKLLMParam()
        
        # Explicitly set all fields to safe values
        param.model_path = None
        param.max_context_len = 0
        param.max_new_tokens = 0
        param.skip_special_token = 0
        param.is_async = 0
        param.img_start = None
        param.img_end = None
        param.img_role = None
        
        print("✅ RKLLMParam structure initialized")
        return param
        
    def _init_extended_param_structure(self) -> RKLLMExtendParam:
        """Safely initialize extended parameter structure"""
        print("Initializing RKLLMExtendParam structure...")
        
        # Create and zero-initialize the structure  
        extend_param = RKLLMExtendParam()
        
        # Explicitly set all fields to safe values
        extend_param.base_domain_id = 0
        extend_param.reserved = None
        
        print("✅ RKLLMExtendParam structure initialized")
        return extend_param
        
    def init_model(self) -> bool:
        """Initialize the RKLLM model with comprehensive error checking"""
        try:
            print("🔧 Starting QwenVL model initialization...")
            
            # Step 1: Validate model file
            if not self._validate_model_file():
                return False
                
            # Step 2: Check if rkllm_lib is available
            if not rkllm_lib:
                print("❌ RKLLM library not available")
                return False
                
            print("✅ RKLLM library available")
            
            # Step 3: Initialize parameter structures safely
            print("Setting up parameters...")
            param = self._init_param_structure()
            
            # Set model path as bytes
            model_path_bytes = self.model_path.encode('utf-8')
            param.model_path = c_char_p(model_path_bytes)
            param.max_context_len = self.max_context_len
            param.max_new_tokens = self.max_new_tokens
            param.skip_special_token = 1
            param.is_async = 0
            
            # Set image tokens (if needed)
            img_start_bytes = b"<image>"
            img_end_bytes = b"</image>"  
            img_role_bytes = b"user"
            
            param.img_start = c_char_p(img_start_bytes)
            param.img_end = c_char_p(img_end_bytes)
            param.img_role = c_char_p(img_role_bytes)
            
            print("✅ Parameters set")
            
            # Step 4: Initialize extended parameters
            print("Setting up extended parameters...")
            extend_param = self._init_extended_param_structure()
            extend_param.base_domain_id = 0
            
            print("✅ Extended parameters set")
            
            # Step 5: Validate structure sizes
            param_size = sizeof(param)
            extend_param_size = sizeof(extend_param)
            print(f"Parameter structure size: {param_size} bytes")
            print(f"Extended parameter structure size: {extend_param_size} bytes")
            
            # Step 6: Initialize handle
            self.handle = c_void_p()
            
            # Step 7: Call rkllm_init with error checking
            print("Calling rkllm_init...")
            print(f"Model path: {self.model_path}")
            print(f"Max context length: {self.max_context_len}")
            print(f"Max new tokens: {self.max_new_tokens}")
            
            # Add a small delay to ensure everything is ready
            import time
            time.sleep(0.1)
            
            try:
                ret = rkllm_lib.rkllm_init(byref(self.handle), byref(param), byref(extend_param))
                print(f"rkllm_init returned: {ret}")
                
                if ret != 0:
                    print(f"❌ rkllm_init failed with return code: {ret}")
                    return False
                    
            except Exception as e:
                print(f"❌ Exception during rkllm_init: {e}")
                return False
            
            # Step 8: Verify handle was created
            if not self.handle or not self.handle.value:
                print("❌ rkllm_init returned success but handle is null")
                return False
                
            print(f"✅ rkllm_init successful, handle: {self.handle.value}")
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"❌ QwenVL model initialization failed: {e}")
            logger.error(f"QwenVL model initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def generate_response(self, prompt: str, image_embeddings: Optional[np.ndarray] = None) -> str:
        """Generate response from the model"""
        if not self.is_initialized:
            print("❌ Model not initialized")
            return "Error: Model not initialized"
            
        try:
            print(f"🔄 Generating response for prompt: {prompt[:50]}...")
            
            # For now, return a placeholder since we're focusing on initialization
            return "Model initialized successfully - response generation not yet implemented"
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"Error: {e}"
            
    def release(self):
        """Release model resources"""
        if self.is_initialized and self.handle:
            try:
                print("🔧 Releasing RKLLM model...")
                ret = rkllm_lib.rkllm_destroy(self.handle)
                if ret == 0:
                    print("✅ RKLLM model released successfully")
                else:
                    print(f"⚠️ RKLLM destroy returned: {ret}")
                    
                self.handle = None
                self.is_initialized = False
                
            except Exception as e:
                print(f"❌ Error releasing model: {e}")
                logger.error(f"Error releasing model: {e}")