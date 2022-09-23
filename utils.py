import random
import numpy as np
import torch
import torch.nn as nn

def seed_all(seed):
    # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def rms(x):
    # x: (B x L)
    L = x.shape[1]
    return (torch.sum(x**2, dim=1) / L)**(1/2)

def rms_mae(a, b):
    # a, b: (B x L)
    return abs(rms(a) - rms(b)).mean()

def load_model(model, path):
    pass

if __name__ == "__main__":
    from src.data.dataset import NonContactTHD
    from src.model.net import ConvNet, NeuralNet, Demo
    import src.model.ts as ts

    seed_all(42)

    model = Demo(200, 64, 200, 128)
    model.load_state_dict(torch.load(f"./model_64_0.025_8000_no_norm_v2.pt"))
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