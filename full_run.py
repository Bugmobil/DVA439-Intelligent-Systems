import numpy as np
import requests
import torch
from PIL import Image
import torchvision.transforms.functional as F
import qai_hub as hub
import os
import importlib
from quantization import quantization_job, compile_quantized_model



modules = ["ChaIR.model_ots_4073_9968", "FocalNet.ots", "IRNeXt.ots", "CasDyF-Net.OTS", "DSANet.ots", "SANet.ots", "FocalNet.model_ots", "OKNet.ots", "C2PNet."]
for i, item in enumerate(modules, 1):
  module_name = item.split('.')[0]
  print(f"{i}. {module_name}")

option = int(input("** Available models ** \n1. Chair\n2. FocalNet\n3. IRNeXt\nChoose Model: "))

# Use the model definition from downloaded repo
# Should look like this: model_repos.<repo_name>.<model_name>.<model_file>.build_net.<ots_name>

module_name, ots_name = modules[option - 1].split('.')
module_path, func_name = f"model_repos.{module_name}.Dehazing.OTS.models.{module_name}.build_net".rsplit('.', 1)
module_name = module_path.split('.')[1]
module = importlib.import_module(module_path)
build_net = getattr(module, func_name)


# 1. Instantiate model and load the state dictionary:
if module_name == "ConvIR":
    model = build_net(version="base")
else:
    model = build_net()
# Load the state dictionary from the pretrained_models folder
state_dict = torch.load(f"pretrained_models/pkl/{module_name}/{ots_name}.pkl", map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)
model.eval() 
print("Instantiation complete")


# Use the model definition from downloaded repo
# Should look like this: model_repos.<repo_name>.<model_name>.<model_file>
#from model_repos.ChaIR.Dehazing.OTS.models.ChaIR import build_net
#model = build_net()

# Load the state dictionary from the pretrained_models folder
#pkl_file_path =  "/ChaIR/model_ots_4073_9968.pkl" # Path to your .pkl file
#final_pkl_path = pkl_base_path + pkl_file_path

# 1. Instantiate model and load the state dictionary:
# Load the state dictionary from the pretrained_models folder
#state_dict = torch.load(final_pkl_path , map_location=torch.device('cpu'))
#model.load_state_dict(state_dict, strict=False)
#model.eval() 

# Step 2: Trace the model
input_shape = (1, 3, 256, 256) # Input shape of the model (batch_size, channels, height, width)
example_input = torch.rand(input_shape)
#traced_model = torch.jit.trace(model, example_input)
traced_model = torch.jit.script(model)

# Step 3: Compile the model
compile_onnx_job = hub.submit_compile_job(
    model=traced_model,
    device=hub.Device("QCS8550 (Proxy)"),
    input_specs=dict(image=input_shape),
    options="--target_runtime onnx",
)
assert isinstance(compile_onnx_job, hub.CompileJob)


# Step 4: Profile on cloud-hosted device
target_model = compile_onnx_job.get_target_model()
assert isinstance(target_model, hub.Model)
profile_job = hub.submit_profile_job(
    model=target_model,
    device=hub.Device("QCS8550 (Proxy)"),
)
assert isinstance(profile_job, hub.ProfileJob)

# Step 5: Run inference on cloud-hosted device
sample_image = Image.open(r"datasets\reside-outdoor\test\hazy\0007_0.9_0.16.jpg").resize((256, 256))
input_array = np.transpose(np.array(sample_image, dtype=np.float32) / 255.0, (2, 0, 1))[np.newaxis, ...]

# Run inference using the on-device model on the input image
inference_job = hub.submit_inference_job(
    model=target_model,
    device=hub.Device("QCS8550 (Proxy)"),
    inputs=dict(image=[input_array]),
)
assert isinstance(inference_job, hub.InferenceJob)

on_device_output = inference_job.download_output_data()
assert isinstance(on_device_output, dict)
import cv2 
# Step 6: Post-process the on-device output for an image dehazer model
output_name = list(on_device_output.keys())[2]
dehazed_output = on_device_output[output_name][0]
dehazed_image = np.transpose(dehazed_output[0], (1, 2, 0))
dehazed_image = np.maximum(dehazed_image, 0)
dehazed_image = np.nan_to_num(dehazed_image) 
eps = 1e-8
dehazed_image = np.clip((dehazed_image + eps / 255.0) ** (1/1.7), 0, 1) * 255
dehazed_image = dehazed_image.astype(np.uint8)
job_code = inference_job.job_id
png_file_path = "outputs/png_files/dehazed_image_" + job_code + ".png"
cv2.imwrite(png_file_path, dehazed_image)
