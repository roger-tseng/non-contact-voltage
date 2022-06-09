import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io
from glob import glob
from tqdm import tqdm

from src.data.dataset import NonContactTHD
from src.model.net import ConvNet, NeuralNet
import src.model.ts as ts

seed = 42  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(0)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Data params
sampleLen   = 0.05  # seconds
sampleFreq  = 8000  # Hz
# Model dimensions
hidden_dim  = 64
normalize   = True

model = ts.TS(roll_dim=200, ff_dim=64, n_head=2)
model.load_state_dict(torch.load(f"model_64_0.05_8000.pt"))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()

dev_set = NonContactTHD(mode="dev", sampleLen=sampleLen, sampleFreq=sampleFreq)  # TODO: needs two length params when debug
dev = torch.utils.data.DataLoader(
        dev_set, 32,
        shuffle=False, drop_last=False,
        num_workers=0, pin_memory=True)

dev_mse = 0
for x, y in tqdm(dev):                         # iterate through the dataloader
    x, y = x.float(), y.float()
    if normalize:
        x = nn.functional.normalize(x, dim=1)
    x, y = x.to(device), y.to(device)    # move data to device (cpu/cuda)
    with torch.no_grad():                # disable gradient calculation
        pred = model(x)                  # forward pass (compute output)
        mse_loss = nn.MSELoss()(pred, y)    # compute loss
    dev_mse += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
dev_mse = dev_mse / len(dev.dataset)                    # compute averaged loss

total_error = 0
count = 0
for fname in glob('./data/changed/*.mat'):

    mat = scipy.io.loadmat(fname) # len 1200
    #print(len(mat['valueNoncontVoltage']))
    for start_sample in range(len(mat['valueNoncontVoltage'])-int(sampleLen*sampleFreq)+1):
        Vin = torch.from_numpy(mat['valueNoncontVoltage'])[start_sample:start_sample+int(sampleLen*sampleFreq)].transpose(0,1).float().to(device)
        Vac = torch.from_numpy(mat['valueContactVoltage'])[start_sample:start_sample+int(sampleLen*sampleFreq)].transpose(0,1).float().to(device)

        if normalize:
            Vin = nn.functional.normalize(Vin, dim=1)
        with torch.no_grad():
            Vout = model(Vin)
            error = nn.MSELoss()(Vac, Vout)
            total_error += error.item()
        count += 1
        if start_sample%100==0:
            fig = plt.figure(figsize=(12, 4))
            x_axis = np.linspace(1,int(sampleLen*sampleFreq),num=int(sampleLen*sampleFreq))

            plt.plot(x_axis, Vac.squeeze(dim=0).detach().cpu().numpy(),'b-', label='Vac ref.', linewidth=2)
            plot_factor = 50
            plt.plot(x_axis, plot_factor*Vin.squeeze(dim=0).detach().cpu().numpy(),'r-', label=f'{str(plot_factor)}*Vin input', linewidth=2)
            plt.plot(x_axis, Vout.squeeze(dim=0).detach().cpu().numpy(),'y-', label='model output', linewidth=2)
                
            plt.legend()
            #print(f'{fname.split("/")[-1]} MSE:{error} sample:{start_sample}')
            plt.title(f'{fname.split("/")[-1]} MSE:{error} sample:{start_sample}')
            plt.savefig(f'{fname.split("/")[-1]}.png')
            plt.close()
print()
print(f'Average MSE for synthetic data is: {dev_mse:.4f}')
print(f'Average MSE for actual data is: {total_error/count:.4f}')