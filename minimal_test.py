# minimal_test.py
import sys
import gc
import psutil
import os

def get_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_encoder_only():
    print(f"Starting memory: {get_memory():.1f} MB")
    
    try:
        print("Importing ImageEncoder...")
        from image_encoder import ImageEncoder
        print(f"After import: {get_memory():.1f} MB")
        
        print("Creating ImageEncoder instance...")
        encoder = ImageEncoder("./models/Qwen2-VL-2B-RKLLM/qwen2_vl_2b_vision_rk3588.rknn", 1)
        print(f"After creation: {get_memory():.1f} MB")
        
        print("Initializing encoder...")
        success = encoder.init_encoder()
        print(f"After init: {get_memory():.1f} MB")
        
        if success:
            print("✅ Encoder initialized successfully!")
        else:
            print("❌ Encoder initialization failed")
        
        print("Releasing encoder...")
        encoder.release()
        del encoder
        gc.collect()
        print(f"After cleanup: {get_memory():.1f} MB")
        
        return success
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_llm_only():
    print(f"Starting memory: {get_memory():.1f} MB")
    
    try:
        print("Importing QwenVLModel...")
        from qwen_model import QwenVLModel
        print(f"After import: {get_memory():.1f} MB")
        
        print("Creating QwenVLModel instance...")
        model = QwenVLModel("./models/Qwen2-VL-2B-RKLLM/qwen2-vl-llm_rk3588.rkllm", 256, 1024)
        print(f"After creation: {get_memory():.1f} MB")
        
        print("Initializing model...")
        success = model.init_model()
        print(f"After init: {get_memory():.1f} MB")
        
        if success:
            print("✅ LLM initialized successfully!")
        else:
            print("❌ LLM initialization failed")
        
        print("Releasing model...")
        model.release()
        del model
        gc.collect()
        print(f"After cleanup: {get_memory():.1f} MB")
        
        return success
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing individual model initialization")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "encoder":
            print("Testing Image Encoder only...")
            test_encoder_only()
        elif sys.argv[1] == "llm":
            print("Testing LLM only...")
            test_llm_only()
        else:
            print("Usage: python minimal_test.py [encoder|llm]")
    else:
        print("Testing Image Encoder...")
        encoder_ok = test_encoder_only()
        
        print("\n" + "=" * 50)
        print("Testing LLM Model...")
        llm_ok = test_llm_only()
        
        print("\n" + "=" * 50)
        print("Summary:")
        print(f"Encoder: {'✅ PASS' if encoder_ok else '❌ FAIL'}")
        print(f"LLM: {'✅ PASS' if llm_ok else '❌ FAIL'}")