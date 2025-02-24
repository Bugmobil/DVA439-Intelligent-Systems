import torch
import os
from model_repos.ChaIR.Dehazing.OTS.models.ChaIR import build_net  # Use the model definition from your downloaded repo

pkl_file_path =  r"\ChaIR\model_ots_4073_9968.pkl" # Path to your .pkl file
pkl_base_path = r"pretrained_models\pkl"
final_pkl_path = os.path.join(pkl_base_path, pkl_file_path)
model_name = pkl_file_path.split("\\")[-2]
save_path = r"pretrained_models\pt"
# 1. Instantiate your model and load the state dictionary:
model = build_net()
state_dict = torch.load(pkl_file_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)
model.eval()  # Set to evaluation mode

# 2. Convert the model to TorchScript
# You can choose to script the model if it's supported. Alternatively, if your model has control flow not supported by tracing,
# you might need scripting.
scripted_model = torch.jit.script(model)  # or torch.jit.trace(model, example_input) if tracing is preferred

# 3. Save the scripted model
pt_name = model_name + pkl_file_path.split("\\")[-2].split(".")[0] + ".pt"
scripted_model.save(os.path.join(save_path, pt_name))

print("TorchScript model saved!") 