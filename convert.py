import torch
from model_repos.ChaIR.Dehazing.OTS.models.ChaIR import build_net  # Use the model definition from your downloaded repo

# 1. Instantiate your model and load the state dictionary:
model = build_net()
state_dict = torch.load(r"Models\ChaIR\model_ots_4073_9968.pkl", map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)
model.eval()  # Set to evaluation mode

# 2. Convert the model to TorchScript
# You can choose to script the model if it's supported. Alternatively, if your model has control flow not supported by tracing,
# you might need scripting.
scripted_model = torch.jit.script(model)  # or torch.jit.trace(model, example_input) if tracing is preferred

# 3. Save the scripted model
scripted_model.save(r"qualcomm-AIHub\models\base\ChaIR\scripted_model_ots_4073_9968.pt")

print("TorchScript model saved!") 