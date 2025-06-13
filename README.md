# Qwen2-VL RKLLM Streamlit Web App

This Streamlit web application provides a user-friendly interface for running Qwen2-VL model inference using RKLLM runtime.

## Features

- 🖼️ **Image Upload**: Upload your own images or use the demo image
- 🤖 **Model Selection**: Choose from available models in the `models/` directory
- ⚙️ **Parameter Configuration**: Adjust inference parameters through sliders and dropdowns
- 🚀 **Real-time Inference**: Run inference and see results in the web interface
- 📊 **Output Display**: View both standard output and error messages

## Prerequisites

1. **Built C Application**: Make sure you have successfully built the C application:
   ```bash
   cd app
   ./build.sh
   ```

2. **Python Environment**: Python 3.7 or higher

3. **Models**: Place your model files in the `models/` directory structure:
   ```
   models/
   └── Qwen2-VL-2B-RKLLM/
       ├── qwen2_vl_2b_vision_rk3588.rknn
       └── qwen2-vl-llm_rk3588.rkllm
   ```

4. **Demo Image**: Ensure `data/demo.jpg` exists for the demo functionality

## Installation

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

3. **Configure the app**:
   - Select a model from the sidebar
   - Upload an image or use the demo image
   - Adjust parameters:
     - **Max New Tokens** (default: 128): Maximum tokens to generate
     - **Max Context Length** (default: 512): Maximum input context length
     - **NPU Core Configuration** (default: 3): Number of NPU cores to use

4. **Run inference** by clicking the "🔥 Run Inference" button

## Parameters Explained

### Command Line Arguments
The web app translates the UI inputs into this command:
```bash
./app/build/app <image_path> <vision_model.rknn> <llm_model.rkllm> <max_new_tokens> <max_context_len> <core_num>
```

### Parameter Details

- **Image Path** (arg 1): Path to the input image file
- **Vision Model** (arg 2): Path to the `.rknn` vision encoder model
- **LLM Model** (arg 3): Path to the `.rkllm` language model
- **Max New Tokens** (arg 4): Maximum number of new tokens to generate (1-2048)
- **Max Context Length** (arg 5): Maximum context length for the model (1-4096)
- **Core Number** (arg 6): NPU core configuration:
  - `1`: Single core (AUTO mode)
  - `2`: Dual core (cores 0+1)
  - `3`: Triple core (cores 0+1+2) - **Recommended**

## Troubleshooting

### Common Issues

1. **"App executable not found"**:
   - Make sure you've built the C application: `cd app && ./build.sh`
   - Check that `./app/build/app` exists

2. **"Model files not found"**:
   - Verify your models are in the correct directory structure
   - Ensure both `.rknn` and `.rkllm` files exist

3. **"Models directory not found"**:
   - Create the `models/` directory in the project root
   - Place your model folders inside it

4. **Inference timeout**:
   - The app has a 5-minute timeout for inference
   - For longer inferences, you may need to run the C app directly

5. **Permission errors**:
   - Ensure the `./app/build/app` executable has proper permissions
   - On Linux/macOS: `chmod +x ./app/build/app`

### Performance Tips

- Use **3 NPU cores** for best performance on RK3588
- Adjust **Max New Tokens** based on your needs (lower = faster)
- Keep **Max Context Length** reasonable for your hardware

## File Structure

```
Qwen2-VL-Streamlit-App-with-RKLLM/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README_STREAMLIT.md      # This file
├── app/
│   ├── build/
│   │   └── app              # Built C executable
│   └── ...
├── models/
│   └── Qwen2-VL-2B-RKLLM/
│       ├── *.rknn           # Vision model
│       └── *.rkllm          # Language model
└── data/
    └── demo.jpg             # Demo image
```

## Advanced Usage

### Adding More Models

To add more model variants:

1. Create a new folder in `models/` directory
2. Place the corresponding `.rknn` and `.rkllm` files in the folder
3. The app will automatically detect and list the new model

### Custom Images

Supported image formats:
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)

The app will automatically handle image preprocessing and temporary file management.

## License

This project follows the same license as the main Qwen2-VL RKLLM project.