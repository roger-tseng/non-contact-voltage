import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io
from glob import glob
from tqdm import tqdm

from src.data.dataset import NonContactTHD
from src.model.net import ConvNet, NeuralNet, Demo
import src.model.ts as ts

def seed_all(seed):
    # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_all(42)

model = Demo(200, 64, 200, 128)
model.load_state_dict(torch.load(f"model_64_0.025_8000.pt"))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()

model_weights = {
    "layer1_weight": model.net[0].weight,
    "layer1_bias": model.net[0].bias,
    "layer2_weight": model.net[2].weight,
    "layer2_bias": model.net[2].bias,
    "layer3_weight": model.net[4].weight,
    "layer3_bias": model.net[4].bias,
    "layer4_weight": model.net[6].weight,
    "layer4_bias": model.net[6].bias,
}

for i in model_weights:
    weight = np.float16(model_weights[i].data.detach().cpu().numpy())
    with open(f"./weights/{i}.txt", 'w') as f:
        print("{", file=f, end="")
        flat = weight.flatten()
        print(f'{i}.shape', weight.shape, flat.shape)
        for count, element in enumerate(flat):
            if count!=len(flat)-1:
                print(element, end=", ", file=f)
            else:
                print(element, end="", file=f)
        print("}", file=f)