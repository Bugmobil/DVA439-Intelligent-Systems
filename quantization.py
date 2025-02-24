import os
import numpy as np
import torch
import torchvision
from PIL import Image
import qai_hub as hub

# Use the model definition from downloaded repo
# Should look like this: model_repos.<repo_name>.<model_name>.<model_file>
from model_repos.ChaIR.Dehazing.OTS.models.ChaIR import build_net

