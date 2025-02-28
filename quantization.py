import os
import numpy as np
import torch
import torchvision
from PIL import Image
import qai_hub as hub


def quantization_job(onnx_model_path, input_shape):
    # 1. Load the model
    # Quantize the model
    if not onnx_model_path.endswith(".onnx"):
        unquantized_onnx_model = hub.get_job(onnx_model_path).get_target_model()
        assert isinstance(unquantized_onnx_model, hub.Model)

    # 2. Load and pre-process downloaded calibration data
    sample_inputs = []

    images_dir = "datasets/reside-outdoor/test/hazy/"
    for i, image_path in enumerate(os.listdir(images_dir)):
        if i > 100: # Limit the number of calibration samples
            break
        sample_image = Image.open(os.path.join(images_dir, image_path))
        sample_image = sample_image.convert("RGB").resize(input_shape[2:])
        # Simple normalization to [0,1] range
        sample_input = np.transpose(np.array(sample_image, dtype=np.float32) / 255.0, (2, 0, 1))[np.newaxis, ...]
        
        sample_inputs.append(sample_input)
    calibration_data = dict(image=sample_inputs)

    # 3. Quantize the model
    quantize_job = hub.submit_quantize_job(
        model=unquantized_onnx_model,
        calibration_data=calibration_data,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT8,
    )

    quantized_onnx_model = quantize_job.get_target_model()
    assert isinstance(quantized_onnx_model, hub.Model)
    return quantized_onnx_model

def compile_quantized_model(quantized_onnx_model):
    # 4. Compile to target runtime (TFLite)
    compile_tflite_job = hub.submit_compile_job(
        model=quantized_onnx_model,
        device=hub.Device("QCS8550 (Proxy)"),
        options="--target_runtime tflite --quantize_io",
    )
    assert isinstance(compile_tflite_job, hub.CompileJob)


def main():
    # Load the model
    onnx_model_path = hub.get_job("jp3nmmdx5") # Replace with your job ID
    input_shape = (1, 3, 256, 256)

    # Quantize the model
    quantized_onnx_model = quantization_job(onnx_model_path, input_shape)

    # Compile the quantized model
    compile_quantized_model(quantized_onnx_model)


if __name__ == "__main__":
    main()