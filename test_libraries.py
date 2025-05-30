# test_libraries.py
import os
import sys
import ctypes
import traceback

def test_library_loading():
    """Test if RKNN and RKLLM libraries can be loaded"""
    print("Testing library loading...")
    
    # Check if library files exist
    rknn_lib_path = "./lib/librknnrt.so"
    rkllm_lib_path = "./lib/librkllmrt.so"
    
    print(f"Checking RKNN library: {rknn_lib_path}")
    if os.path.exists(rknn_lib_path):
        print("✅ RKNN library file exists")
    else:
        print("❌ RKNN library file not found")
        return False
    
    print(f"Checking RKLLM library: {rkllm_lib_path}")
    if os.path.exists(rkllm_lib_path):
        print("✅ RKLLM library file exists")
    else:
        print("❌ RKLLM library file not found")
        return False
    
    # Try to load libraries
    try:
        print("Loading RKNN library...")
        rknn_lib = ctypes.CDLL(rknn_lib_path)
        print("✅ RKNN library loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load RKNN library: {e}")
        return False
    
    try:
        print("Loading RKLLM library...")
        rkllm_lib = ctypes.CDLL(rkllm_lib_path)
        print("✅ RKLLM library loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load RKLLM library: {e}")
        return False
    
    return True

def test_model_files():
    """Test if model files exist"""
    print("\nTesting model files...")
    
    encoder_path = "./models/Qwen2-VL-2B-RKLLM/qwen2_vl_2b_vision_rk3588.rknn"
    llm_path = "./models/Qwen2-VL-2B-RKLLM/qwen2-vl-llm_rk3588.rkllm"
    
    print(f"Checking encoder model: {encoder_path}")
    if os.path.exists(encoder_path):
        size = os.path.getsize(encoder_path)
        print(f"✅ Encoder model exists (size: {size:,} bytes)")
    else:
        print("❌ Encoder model not found")
        return False
    
    print(f"Checking LLM model: {llm_path}")
    if os.path.exists(llm_path):
        size = os.path.getsize(llm_path)
        print(f"✅ LLM model exists (size: {size:,} bytes)")
    else:
        print("❌ LLM model not found")
        return False
    
    return True

def test_python_modules():
    """Test if our Python modules can be imported"""
    print("\nTesting Python modules...")
    
    try:
        print("Importing rknn_bindings...")
        import rknn_bindings
        print("✅ rknn_bindings imported successfully")
    except Exception as e:
        print(f"❌ Failed to import rknn_bindings: {e}")
        print(traceback.format_exc())
        return False
    
    try:
        print("Importing image_encoder...")
        from image_encoder import ImageEncoder
        print("✅ image_encoder imported successfully")
    except Exception as e:
        print(f"❌ Failed to import image_encoder: {e}")
        print(traceback.format_exc())
        return False
    
    try:
        print("Importing qwen_model...")
        from qwen_model import QwenVLModel
        print("✅ qwen_model imported successfully")
    except Exception as e:
        print(f"❌ Failed to import qwen_model: {e}")
        print(traceback.format_exc())
        return False
    
    return True

def main():
    print("🔍 RKNN/RKLLM Environment Test")
    print("=" * 50)
    
    # Test library loading
    libs_ok = test_library_loading()
    
    # Test model files
    models_ok = test_model_files()
    
    # Test Python modules (only if libraries are OK)
    if libs_ok:
        modules_ok = test_python_modules()
    else:
        modules_ok = False
        print("\n⚠️ Skipping module tests due to library issues")
    
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"Libraries: {'✅ PASS' if libs_ok else '❌ FAIL'}")
    print(f"Models: {'✅ PASS' if models_ok else '❌ FAIL'}")
    print(f"Modules: {'✅ PASS' if modules_ok else '❌ FAIL'}")
    
    if libs_ok and models_ok and modules_ok:
        print("\n🎉 All tests passed! You can run the Streamlit app.")
    else:
        print("\n⚠️ Some tests failed. Please fix the issues before running the app.")
        
        if not libs_ok:
            print("\n💡 Library troubleshooting:")
            print("1. Make sure you're running on a compatible ARM64 platform (RK3588)")
            print("2. Check if the library files have correct permissions")
            print("3. Install any missing dependencies for RKNN runtime")
        
        if not models_ok:
            print("\n💡 Model troubleshooting:")
            print("1. Download the correct model files")
            print("2. Check the file paths and directory structure")
            print("3. Ensure models are compatible with your hardware")

if __name__ == "__main__":
    main()