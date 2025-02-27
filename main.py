import numpy as np
import requests
import torch
from PIL import Image
import torchvision.transforms.functional as F
import qai_hub as hub
import os



# Use the model definition from downloaded repo
# Should look like this: model_repos.<repo_name>.<model_name>.<model_file>
#from model_repos.SFNet.Image_dehazing.models.SFNet import build_net
#from model_repos.FSNet.Dehazing.OTS.models.FSNet import build_net
#from model_repos.ChaIR.Dehazing.OTS.models.ChaIR import build_net
global model

def trace_model(pkl_file_path):
    # 1. Instantiate model and load the state dictionary:
    # Load the state dictionary from the pretrained_models folder
    state_dict = torch.load(pkl_file_path , map_location=torch.device('cuda:0'))
    model.load_state_dict(state_dict, strict=False)
    model.eval() 

    # Step 2: Trace the model
    input_shape = (1, 3, 256, 256) # Input shape of the model (batch_size, channels, height, width)
    example_input = torch.rand(input_shape)
    traced_model = torch.jit.trace(model, example_input)
    #traced_model = torch.jit.script(model)
    if input("Do you want to save the TorchScript model? (y/n): ") == 'y':
        model_name = pkl_file_path.split('\\')[-2]
        save_path = r"pretrained_models\pt"
        pt_name = model_name + "_" + pkl_file_path.split("\\")[-1].split(".")[0] + ".pt"
        traced_model.save(os.path.join(save_path, pt_name))


def compile_model_qui(file_path):
    #Compile model to ONNX
    
    if file_path.endswith(".pt"):
        compile_type = "--target_runtime onnx"
        input_specs = dict(image=(1, 3, 224, 224))
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

def compile_model_torch(pt_file_path):
    #Compile model to TorchScript
    input_shape = (1, 3, 256, 256) # Input shape of the model (batch_size, channels, height, width)
    example_input = torch.rand(input_shape)
    onnx_name = pt_file_path.split("\\")[-1].split(".")[0] + ".onnx"
    onnx_folder = r"pretrained_models\onnx"
    #pretrained_models\pt\ChaIR_model_ots_4073_9968.pt
    torch.onnx.export(model, example_input, os.path.join(onnx_folder, onnx_name), input_names=["image"], opset_version=11)

def inference_job_qai(job_code):
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
    #Download model
    target_model = hub.get_job(job_code).get_target_model()
    assert isinstance(target_model, hub.Model)
    base_path = "pretrained_models/onnx"
    target_model.download(base_path + "/" + job_code + ".onnx")

def choose_model():
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
    global model
    case = input("** Available Jobs ** \n1. Trace a model\n2. Compile a model\n3. Quantize a ONNX model\n4. Profile a model\n5. Download a model\n6. Do a full run\nChoose Job: ")
    if case == '1':
        model = choose_model()
        pkl_file_path =  input("Enter the path to the .pkl file: ")
        trace_model(pkl_file_path)
    elif case == '2':
        file_path = input("Enter the file path: ")
        if file_path.endswith(".pt"):
            if input("Do you want to compile the model using QUI or TorchScript? (q/t): ") == 't':
                compile_model_torch(file_path)
            else:
                compile_model_qui(file_path)
        else:
            compile_model_qui(file_path)
    elif case == '3':
        from quantization import quantization_job, compile_quantized_model
        #onnx_model_path = input("Enter the path to the ONNX model: ")
        onnx_model_path = None
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
    choose_job()
    
    

if __name__ == "__main__":
    main()
    