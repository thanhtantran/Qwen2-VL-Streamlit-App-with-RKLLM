# minimal_llm_test.py
import os
import sys
import ctypes
from ctypes import *

def test_library_file():
    """Test if librkllmrt.so exists"""
    print("🔍 Testing for librkllmrt.so...")
    
    library_paths = [
        "./lib/librkllmrt.so",
        #"/usr/lib/librkllmrt.so", 
        #"/usr/local/lib/librkllmrt.so",
        #"librkllmrt.so"
    ]
    
    found_lib = None
    for lib_path in library_paths:
        if os.path.exists(lib_path):
            size = os.path.getsize(lib_path)
            print(f"✅ Found library: {lib_path} ({size} bytes)")
            found_lib = lib_path
            break
        else:
            print(f"❌ Not found: {lib_path}")
            
    if not found_lib:
        print("❌ librkllmrt.so not found in any standard location")
        return False
        
    return found_lib

def test_rkllm_bindings():
    """Test if we can create RKLLM bindings"""
    print("🔍 Testing RKLLM bindings creation...")
    
    try:
        # Import our bindings
        from rkllm_bindings import rkllm_lib, RKLLMParam, RKLLMExtendParam
        
        if rkllm_lib:
            print("✅ RKLLM library loaded successfully")
            
            # Test structure creation
            param = RKLLMParam()
            extend_param = RKLLMExtendParam()
            print(f"✅ Structures created - RKLLMParam: {sizeof(param)} bytes, RKLLMExtendParam: {sizeof(extend_param)} bytes")
            
            return True
        else:
            print("❌ RKLLM library is None")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_model_file():
    """Test if model file exists"""
    model_path = "./models/Qwen2-VL-2B-RKLLM/qwen2-vl-llm_rk3588.rkllm"
    print(f"🔍 Testing model file: {model_path}")
    
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        print(f"✅ Model file exists, size: {size:,} bytes ({size/1024/1024:.1f} MB)")
        return True
    else:
        print(f"❌ Model file not found: {model_path}")
        
        # Look for any .rkllm files
        rkllm_files = [f for f in os.listdir('.') if f.endswith('.rkllm')]
        if rkllm_files:
            print("Found other .rkllm files:")
            for f in rkllm_files:
                size = os.path.getsize(f)
                print(f"  - {f} ({size:,} bytes)")
        else:
            print("No .rkllm files found in current directory")
            
        return False

def test_basic_initialization():
    """Test basic RKLLM initialization without segfault"""
    print("🔍 Testing basic RKLLM initialization...")
    
    try:
        from rkllm_bindings import rkllm_lib, RKLLMParam, RKLLMExtendParam
        
        if not rkllm_lib:
            print("❌ RKLLM library not available")
            return False
            
        # Create structures with minimal setup
        param = RKLLMParam()
        extend_param = RKLLMExtendParam()
        handle = c_void_p()
        
        # Initialize all fields to safe values first
        param.model_path = None
        param.max_context_len = 0
        param.max_new_tokens = 0
        param.skip_special_token = 0
        param.is_async = 0
        param.img_start = None
        param.img_end = None
        param.img_role = None
        param.lora_params = None
        param.lora_params_num = 0
        
        extend_param.base_domain_id = 0
        extend_param.reserved = None
        
        print("✅ Structures initialized with safe values")
        
        # Now set actual values
        model_path = "qwen2-vl-2b-instruct.rkllm"
        if not os.path.exists(model_path):
            print(f"❌ Model file required for initialization: {model_path}")
            return False
            
        model_path_bytes = model_path.encode('utf-8')
        param.model_path = c_char_p(model_path_bytes)
        param.max_context_len = 512
        param.max_new_tokens = 512
        param.skip_special_token = 1
        param.is_async = 0
        
        print(f"✅ Parameters set:")
        print(f"   Model path: {model_path}")
        print(f"   Max context: {param.max_context_len}")
        print(f"   Max new tokens: {param.max_new_tokens}")
        
        # Test the dangerous call
        print("⚠️  Calling rkllm_init (this is where segfault might occur)...")
        
        # Add some debug info
        print(f"   Handle address: {id(handle)}")
        print(f"   Param address: {id(param)}")
        print(f"   Extend param address: {id(extend_param)}")
        
        ret = rkllm_lib.rkllm_init(byref(handle), byref(param), byref(extend_param))
        
        print(f"✅ rkllm_init completed with return code: {ret}")
        
        if ret == 0:
            print(f"✅ Initialization successful! Handle: {handle.value}")
            
            # Clean up
            if handle.value:
                print("🔧 Cleaning up...")
                cleanup_ret = rkllm_lib.rkllm_destroy(handle)
                print(f"✅ Cleanup completed with return code: {cleanup_ret}")
                
            return True
        else:
            print(f"❌ Initialization failed with return code: {ret}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during initialization test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Minimal RKLLM Test")
    print("=" * 50)
    
    # Test 1: Check for library file
    lib_path = test_library_file()
    if not lib_path:
        print("Please ensure librkllmrt.so is in the current directory or system path")
        sys.exit(1)
        
    # Test 2: Test bindings creation
    if not test_rkllm_bindings():
        sys.exit(1)
        
    # Test 3: Check model file
    if not test_model_file():
        print("Please ensure the .rkllm model file is available")
        sys.exit(1)
        
    # Test 4: Test basic initialization (the critical test)
    if not test_basic_initialization():
        print("❌ Basic initialization failed - this is where the segfault occurs")
        sys.exit(1)
        
    print("🎉 All tests passed! RKLLM is working correctly.")