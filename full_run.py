import numpy as np
import requests
import torch
from PIL import Image
import torchvision.transforms.functional as F
import qai_hub as hub

# Use the model definition from downloaded repo
# Should look like this: model_repos.<repo_name>.<model_name>.<model_file>
from model_repos.ChaIR.Dehazing.OTS.models.ChaIR import build_net



# 1. Instantiate model and load the state dictionary:
model = build_net()
# Load the state dictionary from the pretrained_models folder
state_dict = torch.load(r"pretrained_models\base\ChaIR\model_ots_4073_9968.pkl", map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)
model.eval() 

# Step 2: Trace the model
input_shape = (1, 3, 256, 256)
example_input = torch.rand(input_shape)
traced_model = torch.jit.trace(model, example_input)

# Step 3: Compile the model
compile_job = hub.submit_compile_job(
    model=traced_model,
    device=hub.Device("QCS8550 (Proxy)"),
    input_specs=dict(image=input_shape),
    options="--target_runtime onnx",
)
assert isinstance(compile_job, hub.CompileJob)

# Step 4: Profile on cloud-hosted device
target_model = compile_job.get_target_model()
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

"""
on_device_output = inference_job.download_output_data()
assert isinstance(on_device_output, dict)

# Step 6: Post-process the on-device output for an image dehazer model
output_name = list(on_device_output.keys())[0]
dehazed_output = on_device_output[output_name][0]
dehazed_image = np.transpose(dehazed_output, (1, 2, 0))
dehazed_image = np.clip(dehazed_image*255, 0, 255).astype(np.uint8)
Image.fromarray(dehazed_image).save("dehazed_image.jpg")


# Step 7: Download model
target_model = compile_job.get_target_model()
assert isinstance(target_model, hub.Model)
target_model.download("ChaIR_ots_4073_9968.onnx")
"""