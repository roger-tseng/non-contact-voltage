import numpy as np
import math

import torch
from torch.utils.data import Dataset

from .generator import ejNCVdataGenerator

class NonContactTHD(Dataset):
    def __init__(self, seed=42, mode='train', sampleLen = 0.05, sampleFreq = 16000, Vamp=110, Vamp_n=0.1, THD=0.1, f=60, f_n=0.5, order=4, debug=False):
        super(NonContactTHD, self).__init__()
        np.random.seed(seed)
        self.debug = debug
        self.mode = mode
        self.Vamp_rms = np.random.uniform(Vamp*(1-Vamp_n), Vamp*(1+Vamp_n), size=len(self))
        self.VthdMax = np.random.uniform(0, THD, size=len(self))
        self.f_fund = np.random.uniform(f-f_n, f+f_n, size=len(self))
        self.phase = np.random.uniform(-math.pi, math.pi, size=(len(self),order+1))
        self.order = order

        mul = 10
        if self.mode in ['dev','test']:
            self.C1 = 18.0E-12 * 10 ** np.linspace(np.log10(1/mul), np.log10(mul), len(self))
            self.C2 = 19.0E-12 * 10 ** np.linspace(np.log10(1/mul), np.log10(mul), len(self))
        else:
            self.C1 = 18.0E-12 * 10 ** np.random.uniform(np.log10(1/mul), np.log10(mul), len(self))
            self.C2 = 19.0E-12 * 10 ** np.random.uniform(np.log10(1/mul), np.log10(mul), len(self))
        
        if self.debug=="shift":
            self.sampleLen = 1
            self.t_start = np.random.randint(0, self.sampleLen*sampleFreq-int(sampleLen*sampleFreq), size=len(self))
            self.start = np.zeros(len(self))
        else:
            self.sampleLen = sampleLen
            self.start = np.random.uniform(0, 1, size=len(self))
        self.sampleFreq = sampleFreq
        self.output_dim = int(sampleLen*sampleFreq)
        #self.gen = ejNCVdataGenerator()
    
    def __getitem__(self, idx): 
        gen = ejNCVdataGenerator(self.Vamp_rms[idx], self.VthdMax[idx], self.f_fund[idx], self.start[idx], phase=self.phase[idx], harmonicOrder = self.order, sampleLen=self.sampleLen, sampleFreq=self.sampleFreq, C1=self.C1[idx], C2=self.C2[idx], enableVinNoise=not(self.debug))
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