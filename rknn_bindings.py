# rknn_bindings.py
import ctypes
from ctypes import *
import logging

logger = logging.getLogger(__name__)

# Load RKNN and RKLLM libraries
try:
    rknn_lib = ctypes.CDLL("./lib/librknnrt.so")
    rkllm_lib = ctypes.CDLL("./lib/librkllmrt.so")
    logger.info("Successfully loaded RKNN and RKLLM libraries")
except Exception as e:
    logger.error(f"Failed to load libraries: {e}")
    raise

# RKNN Constants
RKNN_SUCC = 0
RKNN_TENSOR_UINT8 = 1
RKNN_TENSOR_NHWC = 0
RKNN_TENSOR_NCHW = 1
RKNN_NPU_CORE_AUTO = 0
RKNN_NPU_CORE_0_1 = 3
RKNN_NPU_CORE_0_1_2 = 7

# RKNN Structures
class RKNNTensorAttr(ctypes.Structure):
    _fields_ = [
        ("index", c_uint32),
        ("n_dims", c_uint32),
        ("dims", c_uint32 * 16),
        ("name", c_char * 256),
        ("n_elems", c_uint32),
        ("size", c_uint32),
        ("fmt", c_int),
        ("type", c_int),
        ("qnt_type", c_int),
        ("zp", c_int8),
        ("scale", c_float)
    ]

class RKNNInputOutputNum(ctypes.Structure):
    _fields_ = [
        ("n_input", c_uint32),
        ("n_output", c_uint32)
    ]

class RKNNInput(ctypes.Structure):
    _fields_ = [
        ("index", c_uint32),
        ("buf", c_void_p),
        ("size", c_uint32),
        ("pass_through", c_uint8),
        ("type", c_int),
        ("fmt", c_int)
    ]

class RKNNOutput(ctypes.Structure):
    _fields_ = [
        ("want_float", c_uint8),
        ("is_prealloc", c_uint8),
        ("index", c_uint32),
        ("buf", c_void_p),
        ("size", c_uint32)
    ]

# RKLLM Constants
RKLLM_RUN_NORMAL = 0
RKLLM_RUN_FINISH = 1
RKLLM_RUN_ERROR = 2
RKLLM_INFER_GENERATE = 0
RKLLM_INPUT_PROMPT = 0
RKLLM_INPUT_MULTIMODAL = 1

# RKLLM Structures
class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", c_int32),
    ]

class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", c_char_p),
        ("max_context_len", c_int32),
        ("max_new_tokens", c_int32),
        ("top_k", c_int32),
        ("top_p", c_float),
        ("temperature", c_float),
        ("repeat_penalty", c_float),
        ("frequency_penalty", c_float),
        ("presence_penalty", c_float),
        ("mirostat", c_int32),
        ("mirostat_tau", c_float),
        ("mirostat_eta", c_float),
        ("skip_special_token", c_bool),
        ("is_async", c_bool),
        ("img_start", c_char_p),
        ("img_end", c_char_p),
        ("img_content", c_char_p),
        ("extend_param", RKLLMExtendParam)
    ]

class RKLLMMultimodalInput(ctypes.Structure):
    _fields_ = [
        ("prompt", c_char_p),
        ("image_embed", POINTER(c_float)),
        ("n_image_tokens", c_size_t),
        ("n_image", c_int32),
        ("image_height", c_int32),
        ("image_width", c_int32)
    ]

class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("input_type", c_int),
        ("prompt_input", c_char_p),
        ("multimodal_input", RKLLMMultimodalInput)
    ]

class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", c_int),
        ("keep_history", c_int)
    ]

class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", c_char_p),
        ("num", c_int)
    ]

# RKNN Function prototypes
rknn_lib.rknn_init.argtypes = [POINTER(c_void_p), c_void_p, c_uint32, c_uint32, c_void_p]
rknn_lib.rknn_init.restype = c_int

rknn_lib.rknn_destroy.argtypes = [c_void_p]
rknn_lib.rknn_destroy.restype = c_int

rknn_lib.rknn_set_core_mask.argtypes = [c_void_p, c_int]
rknn_lib.rknn_set_core_mask.restype = c_int

rknn_lib.rknn_query.argtypes = [c_void_p, c_int, c_void_p, c_uint32]
rknn_lib.rknn_query.restype = c_int

rknn_lib.rknn_inputs_set.argtypes = [c_void_p, c_uint32, POINTER(RKNNInput)]
rknn_lib.rknn_inputs_set.restype = c_int

rknn_lib.rknn_run.argtypes = [c_void_p, c_void_p]
rknn_lib.rknn_run.restype = c_int

rknn_lib.rknn_outputs_get.argtypes = [c_void_p, c_uint32, POINTER(RKNNOutput), c_void_p]
rknn_lib.rknn_outputs_get.restype = c_int

rknn_lib.rknn_outputs_release.argtypes = [c_void_p, c_uint32, POINTER(RKNNOutput)]
rknn_lib.rknn_outputs_release.restype = c_int

# RKLLM Function prototypes
CallbackFunc = ctypes.CFUNCTYPE(None, POINTER(RKLLMResult), c_void_p, c_int)

rkllm_lib.rkllm_init.argtypes = [POINTER(c_void_p), POINTER(RKLLMParam), CallbackFunc]
rkllm_lib.rkllm_init.restype = c_int

rkllm_lib.rkllm_destroy.argtypes = [c_void_p]
rkllm_lib.rkllm_destroy.restype = c_int

rkllm_lib.rkllm_run.argtypes = [c_void_p, POINTER(RKLLMInput), POINTER(RKLLMInferParam), c_void_p]
rkllm_lib.rkllm_run.restype = c_int

rkllm_lib.rkllm_set_chat_template.argtypes = [c_void_p, c_char_p, c_char_p, c_char_p]
rkllm_lib.rkllm_set_chat_template.restype = c_int

rkllm_lib.rkllm_clear_kv_cache.argtypes = [c_void_p, c_int]
rkllm_lib.rkllm_clear_kv_cache.restype = c_int