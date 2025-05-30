# rkllm_bindings.py
import os
import ctypes
from ctypes import *
import logging

logger = logging.getLogger(__name__)

# Load the RKLLM library
rkllm_lib = None

def load_rkllm_library():
    """Load the RKLLM library from librkllmrt.so"""
    global rkllm_lib
    
    library_paths = [
        "./lib/librkllmrt.so",
        #"/usr/lib/librkllmrt.so", 
        #"/usr/local/lib/librkllmrt.so",
        #"librkllmrt.so"
    ]
    
    for lib_path in library_paths:
        try:
            if os.path.exists(lib_path):
                print(f"🔍 Trying to load RKLLM library from: {lib_path}")
                rkllm_lib = ctypes.CDLL(lib_path)
                print(f"✅ Successfully loaded RKLLM library from: {lib_path}")
                return True
            else:
                print(f"❌ Library not found at: {lib_path}")
        except Exception as e:
            print(f"❌ Failed to load library from {lib_path}: {e}")
            continue
            
    print("❌ Could not load RKLLM library from any path")
    return False

# Load the library
if not load_rkllm_library():
    print("⚠️ RKLLM library not loaded - functions will not be available")

# Define RKLLM enums and constants
class RKLLMInferenceType(ctypes.c_int):
    RKLLM_INFER_GENERATE = 0
    RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1

class RKLLMTokenizerType(ctypes.c_int):
    RKLLM_TOKENIZER_AUTO = 0
    RKLLM_TOKENIZER_TIKTOKEN = 1
    RKLLM_TOKENIZER_SENTENCEPIECE = 2

# Constants
RKLLM_RUN_NORMAL = 0
RKLLM_RUN_ASYNC = 1

# Define RKLLM structures
class RKLLMLoraParam(Structure):
    _fields_ = [
        ("lora_model_path", c_char_p),
        ("lora_adapter_name", c_char_p),
        ("scale", c_float),
    ]

class RKLLMParam(Structure):
    _fields_ = [
        ("model_path", c_char_p),
        ("max_context_len", c_int32),
        ("max_new_tokens", c_int32),
        ("skip_special_token", c_int32),
        ("is_async", c_int32),
        ("img_start", c_char_p),
        ("img_end", c_char_p), 
        ("img_role", c_char_p),
        ("lora_params", POINTER(RKLLMLoraParam)),
        ("lora_params_num", c_int32),
    ]

class RKLLMExtendParam(Structure):
    _fields_ = [
        ("base_domain_id", c_uint32),
        ("reserved", c_char_p),
    ]

class RKLLMInputEmbed(Structure):
    _fields_ = [
        ("embed", POINTER(c_float)),
        ("n_tokens", c_size_t),
    ]

class RKLLMInput(Structure):
    _fields_ = [
        ("input_mode", c_int32),  # 0 for text, 1 for embed
        ("input_data", c_void_p), # char* for text, RKLLMInputEmbed* for embed
    ]

class RKLLMResult(Structure):
    _fields_ = [
        ("text", c_char_p),
        ("size", c_int32),
    ]

# Define function prototypes if library is loaded
if rkllm_lib:
    try:
        # rkllm_init
        rkllm_lib.rkllm_init.argtypes = [POINTER(c_void_p), POINTER(RKLLMParam), POINTER(RKLLMExtendParam)]
        rkllm_lib.rkllm_init.restype = c_int32
        
        # rkllm_run
        rkllm_lib.rkllm_run.argtypes = [c_void_p, POINTER(RKLLMInput), POINTER(RKLLMResult)]
        rkllm_lib.rkllm_run.restype = c_int32
        
        # rkllm_destroy
        rkllm_lib.rkllm_destroy.argtypes = [c_void_p]
        rkllm_lib.rkllm_destroy.restype = c_int32
        
        # rkllm_abort
        rkllm_lib.rkllm_abort.argtypes = [c_void_p]
        rkllm_lib.rkllm_abort.restype = c_int32
        
        print("✅ RKLLM function prototypes defined successfully")
        
    except Exception as e:
        print(f"❌ Error defining function prototypes: {e}")
        rkllm_lib = None
else:
    print("⚠️ Skipping function prototype definition - library not loaded")

# Export the library and structures
__all__ = [
    'rkllm_lib',
    'RKLLMParam', 
    'RKLLMExtendParam',
    'RKLLMLoraParam',
    'RKLLMInput',
    'RKLLMInputEmbed', 
    'RKLLMResult',
    'RKLLMInferenceType',
    'RKLLMTokenizerType',
    'RKLLM_RUN_NORMAL',
    'RKLLM_RUN_ASYNC'
]