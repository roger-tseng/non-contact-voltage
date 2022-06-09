from src.data.dataset import NonContactTHD
from src.model.net import ConvNet, NeuralNet, Demo
import src.model.ts as ts

import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

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
sampleFreq  = 8000#, 8000, 4000, 2000]  # Hz
#sampleFreqs  = [1000, 2000, 4000, 8000, 16000]  # Hz

# Model dimensions
hidden_dim  = 64#, 16, 8]
#hidden_dims  = [1, 2, 4, 8, 16, 32, 64, 128]

# Training params
batch_size = 32
workers = 0
epochs = 300
early_stop = 15
normalize = True  
do_early_stop = True





samples = int(sampleLen*sampleFreq)
print("----------------------------------------------------------------")
print(f"Measuring {sampleLen} secs at {sampleFreq} Hz for {samples} samples...")
print()
train_set = NonContactTHD(mode="train", sampleLen=sampleLen, sampleFreq=sampleFreq)  # Construct dataset
dev_set = NonContactTHD(mode="dev", sampleLen=sampleLen, sampleFreq=sampleFreq)  # TODO: needs two length params when debug
test_set = NonContactTHD(mode="test", sampleLen=sampleLen, sampleFreq=sampleFreq)

# Construct dataloader
train = DataLoader(
        train_set, batch_size,
        shuffle=False, drop_last=False,
        num_workers=workers, pin_memory=True)
dev = DataLoader(
        dev_set, batch_size,
        shuffle=False, drop_last=False,
        num_workers=workers, pin_memory=True)
test = DataLoader(
        test_set, 1,
        shuffle=False, drop_last=False,
        num_workers=workers, pin_memory=True)

idx = 1
Vin_raw, Vac_raw = dev_set[idx]
if dev_set.debug=="shift":
    t = dev_set.t_start[idx]
    Vin = Vin_raw[t:t+samples]
    Vac = Vac_raw[t:t+samples]
else:
    Vin = Vin_raw
    Vac = Vac_raw

# fig = plt.figure(figsize=(12, 4))
# x_axis = np.linspace(1,samples,num=samples)
# # draw AC voltage and input voltage
# plt.plot(x_axis, Vac.detach().numpy(),'b-', label='Vac ref.', linewidth=2)
# plt.plot(x_axis, 30*Vin.detach().numpy(),'r-', label='30*Vin input', linewidth=2)
# plt.legend()
# plt.show()
print("----------------------------------------------------------------")

#plt.savefig('books_read.png')
print(f'\n Using {hidden_dim} dimensions... \n')

model = Demo(200, 64, 200, 128)
#model = ts.TS(roll_dim=200, ff_dim=64, n_head=2)
#model = ConvNet(samples, hidden_dim, samples)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
criterion = nn.MSELoss()

min_mse = 10000.
loss_record = {'train': [], 'dev': []}      # for recording training loss
early_stop_cnt = 0

for epoch in range(epochs):
    model.train()                           # set model to training mode
    train_loss = 0
    for x, y in tqdm(train):                # iterate through the dataloader
        x, y = x.float(), y.float()
        #if torch.isnan(x).any():
        #    print('x is nan!\n\n')
        #    break
        #if torch.isnan(y).any():
        #    print('y is nan!\n\n')
        #    break
        optimizer.zero_grad()               # set gradient to zero
        if normalize:
            x = nn.functional.normalize(x, dim=1)
        x, y = x.to(device), y.to(device)   # move data to device (cpu/cuda)
        pred = model(x)                     # forward pass (compute output)
        #if torch.isnan(pred).any():
        #    print('pred is nan!\n\n')
        #    break
        mse_loss = criterion(pred, y)       # compute loss
        mse_loss.backward()                 # compute gradient (backpropagation)
        optimizer.step()                    # update model with optimizer
        train_loss += mse_loss.detach().cpu().item() * len(x)
        # with torch.no_grad():
        #     if min_mse<60 and mse_loss > 400:
        #         fig = plt.figure(figsize=(12, 4))
        #         plt.plot(x_axis, y[0].detach().cpu().numpy(),'b-', label='Vac ref.', linewidth=2)
        #         plt.plot(x_axis, 30*x[0].detach().cpu().numpy(),'r-', label='30*Vin input', linewidth=2)
        #         plt.plot(x_axis, pred[0].detach().cpu().numpy(),'g-', label='model output', linewidth=2)
        #         plt.legend()
        #         plt.show()
    train_loss = train_loss / len(train.dataset)
    #loss_record['train'].append(train_loss)

    # After each epoch, test model on the validation (development) set.
    model.eval()
    dev_mse = 0
    for x, y in dev:                         # iterate through the dataloader
        x, y = x.float(), y.float()
        if normalize:
            x = nn.functional.normalize(x, dim=1)
        x, y = x.to(device), y.to(device)    # move data to device (cpu/cuda)
        with torch.no_grad():                # disable gradient calculation
            pred = model(x)                  # forward pass (compute output)
            mse_loss = criterion(pred, y)    # compute loss
        dev_mse += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
    dev_mse = dev_mse / len(dev.dataset)                    # compute averaged loss

    # draw AC voltage and input voltage
    # if epoch%10==5:
    #     fig = plt.figure(figsize=(12, 4))
    #     plt.plot(x_axis, y[0].detach().cpu().numpy(),'b-', label='Vac ref.', linewidth=2)
    #     plt.plot(x_axis, 30*x[0].detach().cpu().numpy(),'r-', label='30*Vin input', linewidth=2)
    #     plt.plot(x_axis, pred[0].detach().cpu().numpy(),'g-', label='model output', linewidth=2)
    #     plt.legend()
    #     plt.show()
        
    print(f'Epoch {(epoch+1):4d}: train loss = {train_loss:.4f}, dev loss = {dev_mse:.4f}')
    if dev_mse < min_mse:
        # Save model if improved
        min_mse = dev_mse
        print('Saving model...')
        torch.save(model.state_dict(), f"./model_{hidden_dim}_{sampleLen}_{sampleFreq}.pt")  # Save model to specified path
        early_stop_cnt = 0
    else:
        early_stop_cnt += 1

    epoch += 1
    #loss_record['dev'].append(dev_mse)
    if do_early_stop and early_stop_cnt > early_stop:
        # Stop training if model stops improving for "config['early_stop']" epochs.
        break

print('Finished training after {} epochs'.format(epoch))
#del model
#!cp 'model_'$hidden_dim'_'$sampleLen'_'$sampleFreq'.pt' '/content/drive/MyDrive/Colab Notebooks/Non-Contact Voltage Measurement/model_variableC1C2_normalize_4layer'$normalize'_'$hidden_dim'_'$sampleLen'_'$sampleFreq'.pt'
#del train_set, dev_set, test_set, train, dev, test  