import numpy as np
import requests
import torch
from PIL import Image
import torchvision.transforms.functional as F
import qai_hub as hub
import os

# Global variable to store the selected model
global model

def trace_model(pkl_file_path):
    """
    Converts a PyTorch model from .pkl format to TorchScript format.
    Args:
        pkl_file_path: Path to the .pkl model file
    Returns:
        traced_model: The traced TorchScript model
    """
    # 1. Instantiate model and load the state dictionary:
    # Load the state dictionary from the pretrained_models folder
    state_dict = torch.load(pkl_file_path , map_location=torch.device('cuda:0'))
    model.load_state_dict(state_dict, strict=False)
    model.eval() 

    # Step 2: Trace the model
    input_shape = (1, 3, 256, 256) # Input shape of the model (batch_size, channels, height, width)
    #example_input = torch.rand(input_shape)
    traced_model = torch.jit.trace(model)
    #traced_model = torch.jit.script(model)
    if input("Do you want to save the TorchScript model? (y/n): ") == 'y':
        model_name = pkl_file_path.split('\\')[-2]
        save_path = r"pretrained_models\pt"
        pt_name = model_name + "_traced_" + pkl_file_path.split("\\")[-1].split(".")[0] + ".pt"
        traced_model.save(os.path.join(save_path, pt_name))
    return traced_model

def compile_model_qui(file_path):
    """
    Compiles a model using Qualcomm AI Hub.
    Supports both ONNX and TFLite formats.
    Args:
        file_path: Path to the model file (.pt or other format)
    """
    #Compile model to ONNX
    
    if file_path.endswith(".pt"):
        compile_type = "--target_runtime onnx"
        input_specs = dict(image=(1, 3, 256, 256))
    else:
        compile_type = "--target_runtime tflite --quantize_io"
        input_specs = None

    compile_job = hub.submit_compile_job(
        model=file_path,
        device=hub.Device("QCS8550 (Proxy)"),
        options=compile_type,
        input_specs=input_specs
    )
    assert isinstance(compile_job, hub.CompileJob)
    compile_job.wait_until_completed()

def compile_model_torch(pt_file_path):
    """
    Converts a PyTorch model to ONNX format using TorchScript.
    Args:
        pt_file_path: Path to the .pt model file
    """
    #Compile model to TorchScript
    input_shape = (1, 3, 256, 256) # Input shape of the model (batch_size, channels, height, width)
    example_input = torch.rand(input_shape)
    onnx_name = pt_file_path.split("\\")[-1].split(".")[0] + ".onnx"
    onnx_folder = r"pretrained_models\onnx"
    #pretrained_models\pt\ChaIR_model_ots_4073_9968.pt
    torch.onnx.export(model, example_input, os.path.join(onnx_folder, onnx_name), input_names=["image"], opset_version=11)

def inference_job_qai(job_code):
    """
    Runs inference on a sample image using a compiled model on Qualcomm AI Hub.
    Args:
        job_code: The job ID from Qualcomm AI Hub compilation
    """
    # Run inference using the on-device model on the input image
    target_model = hub.get_job(job_code).get_target_model()
    assert isinstance(target_model, hub.Model)
    sample_image_path = input("Sample image path (enter for default): ")
    if  sample_image_path == None:
        sample_image = Image.open(r"datasets\reside-outdoor\test\hazy\0001_0.8_0.2.jpg").resize((256, 256))
    else:
        sample_image = Image.open(sample_image_path).resize((256, 256))
    input_array = np.transpose(np.array(sample_image, dtype=np.float32) / 255.0, (2, 0, 1))[np.newaxis, ...]

    inference_job = hub.submit_inference_job(
    model=target_model,
    device=hub.Device("QCS8550 (Proxy)"),
    inputs=dict(image=[input_array]),
    )
    assert isinstance(inference_job, hub.InferenceJob)

def download_a_model(job_code, model_type):
    """
    Downloads a compiled model from Qualcomm AI Hub.
    Args:
        job_code: The job ID from Qualcomm AI Hub
        model_type: Type/name of the model
    """
    #Download model
    target_model = hub.get_job(job_code).get_target_model()
    assert isinstance(target_model, hub.Model)
    base_path = "pretrained_models/onnx"
    target_model.download(base_path + "/" + job_code + ".onnx")

def choose_model():
    """
    Interactive menu for selecting a dehazing model.
    Available models:
    1. ChaIR - Dehazing model
    2. FSNet - Dehazing model
    3. SFNet - Image dehazing model
    Returns:
        The initialized model object
    """
    case_model = input("** Available models ** \n1. Chair\n2. FSNet\n3. SFNet\nChoose Model:")
    if case_model == '1':
        from model_repos.ChaIR.Dehazing.OTS.models.ChaIR import build_net
        return build_net()
    elif case_model == '2':
        from model_repos.FSNet.Dehazing.OTS.models.FSNet import build_net
        return build_net()
    elif case_model == '3':
        from model_repos.SFNet.Image_dehazing.models.SFNet import build_net
        return build_net("test")
    else:
        print("Invalid input")
        choose_model()


def choose_job():
    """
    Interactive menu for selecting model processing operations.
    Available operations:
    1. Trace - Convert PKL to TorchScript
    2. Compile - Convert to ONNX/TFLite
    3. Quantize - Quantize ONNX model
    4. Profile - Profile model performance
    5. Download - Download compiled model
    6. Full Run - Complete pipeline execution
    """
    global model
    torch_script_model = None
    case = input("** Available Jobs ** \n1. Trace a model\n2. Compile a model\n3. Quantize a ONNX model\n4. Profile a model\n5. Download a model\n6. Do a full run\nChoose Job: ")
    if case == '1':
        model = choose_model()
        pkl_file_path =  input("Enter the path to the .pkl file: ")
        torch_script_model = trace_model(pkl_file_path)
    elif case == '2':
        file_path = input("Enter the file path: ")
        if torch_script_model is not None:
            compile_model_torch(torch_script_model)
        elif file_path.endswith(".pt"):
            if input("Do you want to compile the model using QUI or TorchScript? (q/t): ") == 't':
                compile_model_torch(file_path)
            else:
                compile_model_qui(file_path)
        else:
            compile_model_qui(file_path)
    elif case == '3':
        from quantization import quantization_job, compile_quantized_model
        onnx_model_path = input("Enter path to local ONNX model or Job-ID for QAI model: ")
        input_shape = (1, 3, 256, 256)
        quantized_onnx_model = quantization_job(onnx_model_path, input_shape)
        compile_quantized_model(quantized_onnx_model)
    elif case == '4':
        from profiling import profile_model
        profile_model()
    elif case == '5':
        download_a_model(input("Enter the job code: "), input("Enter the model name: "))
    elif case == '6':
        from full_run import full_run
        full_run()
    else:
        print("Invalid input")


def main():
    """
    Main program loop that provides an interactive interface for:
    - Model selection
    - Model format conversion
    - Model compilation
    - Model quantization
    - Performance profiling
    - Model download
    
    Usage:
    1. First select a job type from the menu
    2. Follow the prompts to provide necessary inputs
    3. Results will be saved in the appropriate folders:
       - TorchScript models: pretrained_models/pt/
       - ONNX models: pretrained_models/onnx/
    """
    # loop to choose the job

    while True:
        choose_job()
    
    

if __name__ == "__main__":
    main()
