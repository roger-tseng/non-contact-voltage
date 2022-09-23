import numpy as np
import math

import torch
from torch.utils.data import Dataset

from .generator import ejNCVdataGenerator

class NonContactTHD(Dataset):
    def __init__(self, seed=42, mode='train', sampleLen = 0.05, sampleFreq = 16000, Vamp=110, Vamp_n=0.1, THD=0.1, f=60, f_n=0.5, order=4, debug=False):
        super(NonContactTHD, self).__init__()
        self.debug = debug
        self.mode = mode
        self.Vamp_rms = np.random.uniform(Vamp*(1-Vamp_n), Vamp*(1+Vamp_n), size=len(self))
        self.VthdMax = np.random.uniform(0, THD, size=len(self))
        self.f_fund = np.random.uniform(f-f_n, f+f_n, size=len(self))
        self.phase = np.random.uniform(-math.pi, math.pi, size=(len(self),order+1))
        self.order = order
        self.enableVinNoise = True

        mul = 10
        if self.mode in ['dev','test']:
            self.C1 = 18.0E-12 * 10 ** np.linspace(np.log10(1/mul), np.log10(mul), len(self))
            self.C2 = 19.0E-12 * 10 ** np.linspace(np.log10(1/mul), np.log10(mul), len(self))
            self.enableVinNoise = False
        else:
            self.C1 = 18.0E-12 * 10 ** np.random.uniform(np.log10(1/mul), np.log10(mul), len(self))
            self.C2 = 19.0E-12 * 10 ** np.random.uniform(np.log10(1/mul), np.log10(mul), len(self))
        
        if self.debug=="shift":
            self.sampleLen = 1
            self.t_start = np.random.randint(0, self.sampleLen*sampleFreq-int(sampleLen*sampleFreq), size=len(self))
            self.start = np.zeros(len(self))
            self.enableVinNoise = False
        else:
            self.sampleLen = sampleLen
            self.start = np.random.uniform(0, 1, size=len(self))
        self.sampleFreq = sampleFreq
        self.output_dim = int(sampleLen*sampleFreq)
        #self.gen = ejNCVdataGenerator()
    
    def __getitem__(self, idx): 
        gen = ejNCVdataGenerator(self.Vamp_rms[idx], self.VthdMax[idx], self.f_fund[idx], self.start[idx], phase=self.phase[idx], harmonicOrder = self.order, sampleLen=self.sampleLen, sampleFreq=self.sampleFreq, C1=self.C1[idx], C2=self.C2[idx], enableVinNoise=self.enableVinNoise)
        gen.generateAllData()
        #gen.plotVacAndVinData()
        if self.mode == 'test':
            gen.checkInitialValue()
        return torch.from_numpy(gen.VinData), torch.from_numpy(gen.VacData)
        
    def __len__(self):
        if self.mode == 'train':
            return int(1E5)
        elif self.mode == 'dev':
            return int(3E4)
        elif self.mode == 'test':
            return int(1E4)

class Probe(NonContactTHD):
    def __getitem__(self, idx):
        gen = ejNCVdataGenerator(self.Vamp_rms[idx], self.VthdMax[idx], self.f_fund[idx], self.start[idx], phase=self.phase[idx], harmonicOrder = self.order, sampleLen=self.sampleLen, sampleFreq=self.sampleFreq, C1=self.C1[idx], C2=self.C2[idx], enableVinNoise=self.enableVinNoise)
        gen.generateAllData()
        Vin, Vac = gen.VinData, gen.VacData
        gen2 = ejNCVdataGenerator(110, 0, 60, 0, phase=self.phase[idx], harmonicOrder = self.order, sampleLen=self.sampleLen, sampleFreq=self.sampleFreq, C1=self.C1[idx], C2=self.C2[idx], enableVinNoise=False)
        gen2.generateAllData()
        Vin2, Vac2 = gen2.VinData, gen2.VacData
        return torch.from_numpy(np.concatenate((Vin2, Vin))), torch.from_numpy(Vac)

class GivenCap(NonContactTHD):
    def __init__(self, mode='train', sampleLen = 0.05, sampleFreq = 16000, noise = 0):
        super(GivenCap, self).__init__(mode=mode, sampleLen=sampleLen, sampleFreq=sampleFreq)
        noise = 1+noise
        self.noise1 = 10 ** np.linspace(np.log10(1/noise), np.log10(noise), len(self))
        self.noise2 = 10 ** np.linspace(np.log10(1/noise), np.log10(noise), len(self))

    def __getitem__(self, idx):
        gen = ejNCVdataGenerator(self.Vamp_rms[idx], self.VthdMax[idx], self.f_fund[idx], self.start[idx], phase=self.phase[idx], harmonicOrder = self.order, sampleLen=self.sampleLen, sampleFreq=self.sampleFreq, C1=self.C1[idx], C2=self.C2[idx], enableVinNoise=self.enableVinNoise)
        gen.generateAllData()
        Vin, Vac = gen.VinData, gen.VacData
        return torch.from_numpy(np.append(Vin, (self.C1[idx]*1E12*self.noise1[idx], self.C2[idx]*1E12*self.noise2[idx]))), torch.from_numpy(Vac)

class Multi(NonContactTHD):
    def __init__(self, seed=42, mode='train', sampleLen = 0.05, sampleFreq = 16000, Vamp=110, Vamp_n=0.1, THD=0.1, f=60, f_n=0.5, order=4, debug=False, k = 5):
        super(Multi, self).__init__(mode=mode, sampleLen=sampleLen, sampleFreq=sampleFreq, Vamp=Vamp, Vamp_n=Vamp_n, THD=THD, f=f, f_n=f_n, order=order, debug=debug)
        self.k = k
        #self.phase = np.random.uniform(-math.pi, math.pi, size=(len(self),self.k*(order+1)))
        #self.start = np.random.uniform(0, 1, size=(len(self),self.k))

    def __getitem__(self, idx):
        
        #gen = ejNCVdataGenerator(self.Vamp_rms[idx], self.VthdMax[idx], self.f_fund[idx], self.start[idx][0], phase=self.phase[idx][:self.order+1], harmonicOrder = self.order, sampleLen=self.sampleLen, sampleFreq=self.sampleFreq, C1=self.C1[idx], C2=self.C2[idx], enableVinNoise=self.enableVinNoise, Rin = 4.0E6, Cin = 39.0E-12)
        gen = ejNCVdataGenerator(self.Vamp_rms[idx], self.VthdMax[idx], self.f_fund[idx], self.start[idx], phase=self.phase[idx], harmonicOrder = self.order, sampleLen=self.sampleLen, sampleFreq=self.sampleFreq, C1=self.C1[idx], C2=self.C2[idx], enableVinNoise=self.enableVinNoise, Rin = 4.0E6, Cin = 39.0E-12)
        gen.generateAllData()
        Vin, Vac = gen.VinData, gen.VacData
        Vin_signals = [Vin]
        #Vac_signals = [Vac]
        for i in range(1,self.k):
            #gen2 = ejNCVdataGenerator(self.Vamp_rms[idx], self.VthdMax[idx], self.f_fund[idx], self.start[idx][i], phase=self.phase[idx][i*(self.order+1):(i+1)*(self.order+1)], harmonicOrder = self.order, sampleLen=self.sampleLen, sampleFreq=self.sampleFreq, C1=self.C1[idx], C2=self.C2[idx], enableVinNoise=self.enableVinNoise, Rin = 4.0E6*(10**i), Cin = 39.0E-12*(10**i))
            gen2 = ejNCVdataGenerator(self.Vamp_rms[idx], self.VthdMax[idx], self.f_fund[idx], self.start[idx], phase=self.phase[idx], harmonicOrder = self.order, sampleLen=self.sampleLen, sampleFreq=self.sampleFreq, C1=self.C1[idx], C2=self.C2[idx], enableVinNoise=self.enableVinNoise, Rin = 4.0E6*(1.125*i), Cin = 39.0E-12*(2*i))
            gen2.harmoAmpRms = gen.harmoAmpRms[1:]
            gen2.generateAllData()
            Vin_signals.append(gen2.VinData)
            #Vac_signals.append(gen2.VacData)
        
        return torch.from_numpy(np.concatenate(Vin_signals)), torch.from_numpy(Vac)
    
    def __len__(self):
        if self.mode == 'train':
            return int(1E5)
        elif self.mode == 'dev':
            return int(3E4)
        elif self.mode == 'test':
            return int(1E4)