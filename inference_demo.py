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

seed = 42  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(0)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Data params
sampleLen   = 0.025  # seconds
sampleFreq  = 8000  # Hz
# Model dimensions
hidden_dim  = 64
normalize   = False

model = Demo(200, 64, 200, 128)
model.load_state_dict(torch.load(f"./model_64_0.025_8000_no_norm_v4.pt"))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()

dev_set = NonContactTHD(mode="dev", sampleLen=sampleLen, sampleFreq=sampleFreq)  # TODO: needs two length params when debug
dev = torch.utils.data.DataLoader(
        dev_set, 1,
        shuffle=False, drop_last=False,
        num_workers=0, pin_memory=True)

dev_mse = 0
dev_rms = 0
avg_rms = 0
rms_list = []
for i, (x, y) in enumerate(tqdm(dev)):                         # iterate through the dataloader
    x, y = x.float(), y.float()
    if normalize:
        x = nn.functional.normalize(x, dim=1)
    x, y = x.to(device), y.to(device)    # move data to device (cpu/cuda)
    Vac_rms = ((sum(y.squeeze()**2)/len(y.squeeze()))**(1/2)).item()
    with torch.no_grad():                # disable gradient calculation
        pred = model(x) #pred = torch.cat((model(nn.functional.normalize(x[:,:200], dim=1)), model(nn.functional.normalize(x[:,200:400], dim=1)), model(nn.functional.normalize(x[:,400:600], dim=1)), model(nn.functional.normalize(x[:,600:], dim=1))), dim=1)                  # forward pass (compute output)
        Vout_rms = ((sum(pred.squeeze()**2)/len(pred.squeeze()))**(1/2)).item()
        #print(Vout_rms, Vac_rms)
        mse_loss = nn.MSELoss()(pred, y)    # compute loss
    avg_rms += abs(Vac_rms - Vout_rms)
    dev_rms += abs(Vac_rms - Vout_rms)
    dev_mse += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
    if (i+1)%100==0:
        rms_list.append(avg_rms/100)
        avg_rms = 0
dev_mse = dev_mse / len(dev.dataset)                    # compute averaged loss
dev_rms = dev_rms / len(dev.dataset)

plt.figure(figsize=(12,4), dpi= 300, facecolor='w', edgecolor='k')
plt.plot(10 ** np.linspace(np.log10(1/10), np.log10(10), len(rms_list)), rms_list)
plt.xlabel("Ratio of capacitance to original values")
plt.ylabel("RMS")
plt.xscale("log")
#plt.title("Small THD")
plt.savefig("rms.png")
plt.close()

total_error = 0
rms_error = 0
count = 0
for fname in tqdm(glob('./data/changed/*.mat')):

    mat = scipy.io.loadmat(fname) # len 1200
    #print(len(mat['valueNoncontVoltage']))
    for start_sample in range(len(mat['valueNoncontVoltage'])-(int(sampleLen*sampleFreq)+1)*2):
        Vin = torch.from_numpy(mat['valueNoncontVoltage'])[start_sample:start_sample+int(sampleLen*sampleFreq)*2:2].transpose(0,1).float().to(device)
        Vac = torch.from_numpy(mat['valueContactVoltage'])[start_sample:start_sample+int(sampleLen*sampleFreq)*2:2].transpose(0,1).float().to(device)
        Vac_rms = ((sum(Vac.squeeze()**2)/len(Vac.squeeze()))**(1/2)).item()

        if normalize:
            Vin = nn.functional.normalize(Vin, dim=1)
        with torch.no_grad():
            Vout = model(Vin) #Vout = torch.cat((model(nn.functional.normalize(Vin[:,:200], dim=1)), model(nn.functional.normalize(Vin[:,200:400], dim=1)), model(nn.functional.normalize(Vin[:,400:600], dim=1)), model(nn.functional.normalize(Vin[:,600:], dim=1))), dim=1)
            Vout_rms = ((sum(Vout.squeeze()**2)/len(Vout.squeeze()))**(1/2)).item()
            rms_diff = abs(Vac_rms - Vout_rms)
            error = nn.MSELoss()(Vac, Vout)
            total_error += error.item()
            rms_error += rms_diff
        count += 1
        #print(Vout_rms, Vac_rms)
        if start_sample%100==0:
            fig = plt.figure(figsize=(12, 4))
            x_axis = np.linspace(1,int(sampleLen*sampleFreq),num=int(sampleLen*sampleFreq))

            plt.plot(x_axis, Vac.squeeze(dim=0).detach().cpu().numpy(),'b-', label='Vac ref.', linewidth=2)
            plot_factor = 50
            plt.plot(x_axis, plot_factor*Vin.squeeze(dim=0).detach().cpu().numpy(),'r-', label=f'{str(plot_factor)}*Vin input', linewidth=2)
            plt.plot(x_axis, Vout.squeeze(dim=0).detach().cpu().numpy(),'y-', label='model output', linewidth=2)
                
            plt.legend()
            #print(f'{fname.split("/")[-1]} MSE:{error} sample:{start_sample}')
            plt.title(f'{fname.split("/")[-1]} MSE:{error} RMS diff:{rms_diff} sample:{start_sample}')
            plt.savefig(f'{fname.split("/")[-1]}.png')
            plt.close()
print()
print(f'Average MSE for synthetic data is: {dev_mse:.4f}')
print(f'Average MAE for synthetic data RMS is: {dev_rms:.4f}')
print(f'Average MSE for actual data is: {total_error/count:.4f}')
print(f'Average MAE for actual data RMS is: {rms_error/count:.4f}')
