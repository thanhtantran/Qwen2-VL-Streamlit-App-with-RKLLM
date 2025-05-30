# image_encoder.py
import cv2
import numpy as np
from PIL import Image
import logging
import ctypes  # Added missing import
from typing import Optional, Tuple
from ctypes import *

from rknn_bindings import (
    rknn_lib, RKNNTensorAttr, RKNNInputOutputNum, RKNNInput, RKNNOutput,
    RKNN_SUCC, RKNN_TENSOR_UINT8, RKNN_TENSOR_NHWC, RKNN_TENSOR_NCHW,
    RKNN_NPU_CORE_AUTO, RKNN_NPU_CORE_0_1, RKNN_NPU_CORE_0_1_2
)

logger = logging.getLogger(__name__)

class ImageEncoder:
    def __init__(self, model_path: str, core_num: int = 1):
        self.model_path = model_path.encode('utf-8')
        self.core_num = core_num
        self.model_height = 392
        self.model_width = 392
        self.model_channel = 6  # 6 channels to match expected input size
        self.image_token_num = 196
        self.embed_size = 1536
        self.rknn_ctx = None
        self.input_attrs = None
        self.output_attrs = None
        self.io_num = RKNNInputOutputNum()
        self.is_initialized = False
        
    def read_model_file(self, path: str) -> bytes:
        with open(path, 'rb') as f:
            return f.read()
        
    def init_encoder(self) -> bool:
        try:
            # Read model
            model_data = self.read_model_file(self.model_path.decode())
            print(f"Model file size: {len(model_data)} bytes")
            
            # Initialize RKNN
            self.rknn_ctx = c_void_p()
            model_buffer = (c_char * len(model_data)).from_buffer_copy(model_data)
            
            print("Calling rknn_init...")
            ret = rknn_lib.rknn_init(byref(self.rknn_ctx), model_buffer, len(model_data), 0, None)
            if ret != RKNN_SUCC:
                raise Exception(f"rknn_init failed with code {ret}")
            print("rknn_init successful")
            
            # Set core mask
            print("Setting core mask...")
            if self.core_num == 2:
                core_mask = RKNN_NPU_CORE_0_1
            elif self.core_num == 3:
                core_mask = RKNN_NPU_CORE_0_1_2
            else:
                core_mask = RKNN_NPU_CORE_AUTO
            
            ret = rknn_lib.rknn_set_core_mask(self.rknn_ctx, core_mask)
            if ret != RKNN_SUCC:
                raise Exception(f"rknn_set_core_mask failed with code {ret}")
            print("Core mask set successfully")
            
            # Query I/O info
            print("Querying I/O info...")
            ret = rknn_lib.rknn_query(self.rknn_ctx, 4, byref(self.io_num), sizeof(self.io_num))
            if ret != RKNN_SUCC:
                print(f"Query I/O failed with code {ret}, using defaults")
                self.io_num.n_input = 1
                self.io_num.n_output = 1
            
            print(f"Input count: {self.io_num.n_input}, Output count: {self.io_num.n_output}")
            
            # Ensure we have at least 1 input and output
            if self.io_num.n_input == 0:
                self.io_num.n_input = 1
            if self.io_num.n_output == 0:
                self.io_num.n_output = 1
            
            # Create input attributes
            print("Setting up input attributes...")
            self.input_attrs = (RKNNTensorAttr * self.io_num.n_input)()
            for i in range(self.io_num.n_input):
                self.input_attrs[i].index = i
                
                # Try to query real attributes
                ret = rknn_lib.rknn_query(self.rknn_ctx, 5, byref(self.input_attrs[i]), sizeof(RKNNTensorAttr))
                if ret == RKNN_SUCC:
                    print(f"Successfully queried input attributes for input {i}")
                    print(f"Input {i} dims: {[self.input_attrs[i].dims[j] for j in range(4)]}")
                    print(f"Input {i} type: {self.input_attrs[i].type}")
                    print(f"Input {i} format: {self.input_attrs[i].fmt}")
                    
                    # Update our dimensions based on actual model
                    if self.input_attrs[i].fmt == RKNN_TENSOR_NCHW:
                        self.model_channel = self.input_attrs[i].dims[1]
                        self.model_height = self.input_attrs[i].dims[2]
                        self.model_width = self.input_attrs[i].dims[3]
                    else:  # NHWC
                        self.model_height = self.input_attrs[i].dims[1]
                        self.model_width = self.input_attrs[i].dims[2]
                        self.model_channel = self.input_attrs[i].dims[3]
                    
                    print(f"Updated model dimensions: {self.model_height}x{self.model_width}x{self.model_channel}")
                else:
                    print(f"Failed to query input attributes for input {i}, using calculated dimensions")
                    # Set to match expected size: 392x392x6 in NHWC format
                    self.input_attrs[i].n_dims = 4
                    self.input_attrs[i].dims[0] = 1  # batch
                    self.input_attrs[i].dims[1] = self.model_height  # height
                    self.input_attrs[i].dims[2] = self.model_width   # width
                    self.input_attrs[i].dims[3] = self.model_channel # channels
                    self.input_attrs[i].type = RKNN_TENSOR_UINT8
                    self.input_attrs[i].fmt = RKNN_TENSOR_NHWC  # Force NHWC format
            
            # Create output attributes
            print("Setting up output attributes...")
            self.output_attrs = (RKNNTensorAttr * self.io_num.n_output)()
            for i in range(self.io_num.n_output):
                self.output_attrs[i].index = i
                ret = rknn_lib.rknn_query(self.rknn_ctx, 6, byref(self.output_attrs[i]), sizeof(RKNNTensorAttr))
                if ret == RKNN_SUCC:
                    print(f"Successfully queried output attributes for output {i}")
                else:
                    print(f"Failed to query output attributes for output {i}, using defaults")
            
            self.is_initialized = True
            print("✅ Image encoder initialized successfully")
            print(f"Expected input size: {self.model_height * self.model_width * self.model_channel} bytes")
            print(f"Input format: {'NHWC' if (self.input_attrs and self.input_attrs[0].fmt == RKNN_TENSOR_NHWC) else 'NCHW'}")
            return True
            
        except Exception as e:
            print(f"❌ ImageEncoder init failed: {e}")
            logger.error(f"ImageEncoder init failed: {e}")
            return False
            
    def expand_to_square(self, img: np.ndarray, background_color: Tuple[int, int, int] = (127, 127, 127)) -> np.ndarray:
        height, width = img.shape[:2]
        if width == height:
            return img.copy()
            
        size = max(width, height)
        result = np.full((size, size, 3), background_color, dtype=img.dtype)
        
        x_offset = (size - width) // 2
        y_offset = (size - height) // 2
        result[y_offset:y_offset + height, x_offset:x_offset + width] = img
        
        return result
        
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        img_array = np.array(image)
        
        # Convert RGBA to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        
        # Make square and resize
        square_img = self.expand_to_square(img_array, (127, 127, 127))
        resized_img = cv2.resize(square_img, (self.model_width, self.model_height), interpolation=cv2.INTER_LINEAR)
        
        # Convert to 6 channels by duplicating RGB channels
        # This creates RGBRGB format to match the expected 921984 bytes
        if self.model_channel == 6:
            print("Converting to 6-channel format (RGBRGB)")
            six_channel_img = np.concatenate([resized_img, resized_img], axis=2)
            processed_img = six_channel_img.astype(np.uint8)
        else:
            processed_img = resized_img.astype(np.uint8)
        
        # Ensure NHWC format (Height, Width, Channels)
        # The data should already be in NHWC format from OpenCV/PIL
        print(f"Final preprocessed shape (NHWC): {processed_img.shape}")
        return processed_img
        
    def encode_image(self, image: Image.Image) -> Optional[np.ndarray]:
        if not self.is_initialized:
            print("❌ Encoder not initialized")
            return None
            
        try:
            print("Preprocessing image...")
            processed_img = self.preprocess_image(image)
            print(f"Processed image shape: {processed_img.shape}")
            
            # Calculate expected size
            expected_size = self.model_height * self.model_width * self.model_channel
            print(f"Expected input size: {expected_size} bytes")
            
            # Prepare input
            inputs = (RKNNInput * 1)()
            inputs[0].index = 0
            inputs[0].type = self.input_attrs[0].type if self.input_attrs else RKNN_TENSOR_UINT8
            inputs[0].fmt = RKNN_TENSOR_NHWC  # Force NHWC format
            inputs[0].size = expected_size
            
            # Flatten the image data (already in NHWC format)
            input_data = processed_img.flatten()
            print(f"Flattened input data size: {len(input_data)} bytes")
            print(f"Input format being used: NHWC")
            
            # Ensure we have exactly the right amount of data
            if len(input_data) != expected_size:
                print(f"❌ Size mismatch: got {len(input_data)}, expected {expected_size}")
                return None
            
            input_buffer = (c_uint8 * len(input_data)).from_buffer_copy(input_data)
            inputs[0].buf = cast(input_buffer, c_void_p)
            
            print("Setting inputs...")
            ret = rknn_lib.rknn_inputs_set(self.rknn_ctx, 1, inputs)
            if ret != RKNN_SUCC:
                raise Exception(f"rknn_inputs_set failed with code {ret}")
            
            print("Running inference...")
            ret = rknn_lib.rknn_run(self.rknn_ctx, None)
            if ret != RKNN_SUCC:
                raise Exception(f"rknn_run failed with code {ret}")
            
            print("Getting outputs...")
            outputs = (RKNNOutput * 1)()
            outputs[0].want_float = 1
            
            ret = rknn_lib.rknn_outputs_get(self.rknn_ctx, 1, outputs, None)
            if ret != RKNN_SUCC:
                raise Exception(f"rknn_outputs_get failed with code {ret}")
            
            # Copy output
            output_size = self.image_token_num * self.embed_size
            output_data = np.zeros(output_size, dtype=np.float32)
            
            # Check if we have enough data
            actual_size = min(outputs[0].size // 4, output_size)  # divide by 4 for float32
            print(f"Expected output size: {output_size}, Actual size: {actual_size}")
            
            ctypes.memmove(output_data.ctypes.data, outputs[0].buf, actual_size * 4)
            
            rknn_lib.rknn_outputs_release(self.rknn_ctx, 1, outputs)
            
            print("✅ Image encoding successful")
            return output_data[:actual_size]
            
        except Exception as e:
            print(f"❌ Image encoding failed: {e}")
            logger.error(f"Image encoding failed: {e}")
            return None
            
    def release(self):
        if self.is_initialized and self.rknn_ctx:
            print("Releasing RKNN context...")
            rknn_lib.rknn_destroy(self.rknn_ctx)
            self.rknn_ctx = None
            self.is_initialized = False
            print("✅ RKNN context released")