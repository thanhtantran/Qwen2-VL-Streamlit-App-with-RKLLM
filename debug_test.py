# debug_test.py
import sys
import gc
import psutil
import os

def get_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_encoder_step_by_step():
    print("🔍 Testing Image Encoder Step by Step")
    print("=" * 50)
    print(f"Starting memory: {get_memory():.1f} MB")
    
    try:
        print("Step 1: Importing ImageEncoder...")
        from image_encoder import ImageEncoder
        print(f"✅ Import successful. Memory: {get_memory():.1f} MB")
        
        print("Step 2: Creating instance...")
        encoder = ImageEncoder("./models/Qwen2-VL-2B-RKLLM/qwen2_vl_2b_vision_rk3588.rknn", 1)
        print(f"✅ Instance created. Memory: {get_memory():.1f} MB")
        
        print("Step 3: Initializing encoder...")
        success = encoder.init_encoder()
        print(f"Result: {'✅ SUCCESS' if success else '❌ FAILED'}. Memory: {get_memory():.1f} MB")
        
        if success:
            print("Step 4: Testing image encoding...")
            from PIL import Image
            import numpy as np
            
            # Create a test image
            test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            print("Created test image")
            
            features = encoder.encode_image(test_img)
            if features is not None:
                print(f"✅ Encoding successful! Features shape: {features.shape}")
            else:
                print("❌ Encoding failed")
        
        print("Step 5: Releasing encoder...")
        encoder.release()
        del encoder
        gc.collect()
        print(f"✅ Released. Memory: {get_memory():.1f} MB")
        
        return success
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_step_by_step():
    print("🔍 Testing LLM Model Step by Step")
    print("=" * 50)
    print(f"Starting memory: {get_memory():.1f} MB")
    
    try:
        print("Step 1: Importing QwenVLModel...")
        from qwen_model import QwenVLModel
        print(f"✅ Import successful. Memory: {get_memory():.1f} MB")
        
        print("Step 2: Creating instance...")
        model = QwenVLModel("./models/Qwen2-VL-2B-RKLLM/qwen2-vl-llm_rk3588.rkllm", 128, 512)
        print(f"✅ Instance created. Memory: {get_memory():.1f} MB")
        
        print("Step 3: Initializing model...")
        success = model.init_model()
        print(f"Result: {'✅ SUCCESS' if success else '❌ FAILED'}. Memory: {get_memory():.1f} MB")
        
        if success:
            print("Step 4: Testing text generation...")
            response = model.generate_response("Hello, how are you?")
            print(f"Response: {response[:100]}...")
        
        print("Step 5: Releasing model...")
        model.release()
        del model
        gc.collect()
        print(f"✅ Released. Memory: {get_memory():.1f} MB")
        
        return success
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "encoder":
            test_encoder_step_by_step()
        elif sys.argv[1] == "llm":
            test_llm_step_by_step()
        else:
            print("Usage: python debug_test.py [encoder|llm]")
    else:
        print("Testing both models with detailed debugging...")
        print()
        
        encoder_ok = test_encoder_step_by_step()
        print("\n" + "=" * 70 + "\n")
        
        if encoder_ok:
            llm_ok = test_llm_step_by_step()
        else:
            print("⚠️ Skipping LLM test due to encoder failure")
            llm_ok = False
        
        print("\n" + "=" * 70)
        print("📋 Final Summary:")
        print(f"Image Encoder: {'✅ PASS' if encoder_ok else '❌ FAIL'}")
        print(f"LLM Model: {'✅ PASS' if llm_ok else '❌ FAIL'}")