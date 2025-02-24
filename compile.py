import numpy as np
import requests
import torch
from PIL import Image
#from torchvision.models import mobilenet_v2
import qai_hub as hub


#Compile model to ONNX
if(input("Do you want to compile the model to ONNX? (y/n): ") == 'y'):
    compile_job = hub.submit_compile_job(
        model=r"pretarained_models\base\scripted_model_ots_4073_9968.pt",
        device=hub.Device("QCS8550 (Proxy)"),
        options="--target_runtime onnx",
        input_specs=dict(image=(1, 3, 224, 224)),
    )
    assert isinstance(compile_job, hub.CompileJob)



