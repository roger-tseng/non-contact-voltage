# Written by Hsueh-Ju, Wu @ NTU EEPRO Lab
# Modified by Yuan Tseng @ NTU SPML Lab

import random
import numpy as np
import math
import matplotlib.pyplot as plt

# Restore Sine Wave Information
class sineWave:
    def __init__(self, amp_rms:float, freq_fund:float, multiple = 1, phase = 0):
        self.amp  = amp_rms * math.sqrt(2)      # Peak Value
        self.freq = freq_fund * multiple        # Frequency
        self.mult = multiple                    # Harmonic Order
        self.phas = phase                       # rad/s

    def checkValue(self):
        # Check Data
        print("Multiple  : {}        , data_type:{}".format(self.mult, type(self.mult)))
        print("Freq  (Hz): {}        , data_type:{}".format(self.freq, type(self.freq)))
        print("Amp    (V): {}        , data_type:{}".format(self.amp, type(self.amp)))  
        print("Phase(deg): {}        , data_type:{}".format(self.rad2deg(), type(self.phas)))
        
    def rad2deg(self):
        return self.phas / math.pi * 180

def initAmpTHDFreq(en):
    # Vac:  110V d = random +-10%
    # Vthd:      d = random 0-10%
    # fac:  60Hz d = random +-0.5
    if en:
        rms = 0.1 
        thd = 0.1
        f = 0.5
    else:
        rms = thd = f = 0

    Vrms = 110 * random.uniform(1-rms, 1+rms)     # Vrms initial Value = 110 Vrms
    Vthd = random.uniform(0, thd)                 # Vthd initial Value = 0   %
    f1   = 60 +  random.uniform(-f, f)            # f1   initial Value = 60  Hz
    
    return Vrms, Vthd, f1

# system transfer
def trans2Vin(Vac: sineWave, Rin: float, Cin: float, C1: float, C2: float) -> sineWave:
    
    w = 2.0 * math.pi * Vac.freq
    
    denReal = C1+C2
    denImag = w*Rin*(Cin*(C1+C2)+C1*C2)
    
    tfAmp = w*Rin*C1*C2 / (denReal**2 + denImag**2)**0.5
    tfPhase = math.atan(denImag / denReal)

    tempVinRms = tfAmp * Vac.amp / math.sqrt(2)
    tempVinPhase = Vac.phas + 0.5*math.pi - tfPhase
    
    return sineWave(tempVinRms, Vac.freq / Vac.mult, Vac.mult, tempVinPhase)

def sineValue2Data(V: sineWave, t: np.array)->np.array:
    return V.amp * np.sin(2 * math.pi * V.freq * t + V.phas)

def harmonicAmpGen(Vamp_rms:float, Vthd_per:float, order: int) -> list:
    ampRmsMax = Vamp_rms * Vthd_per
    harmonicRMS = np.flip(np.sort(np.random.rand(order)))
    harmonicRMS = harmonicRMS * math.sqrt(ampRmsMax**2 / np.sum(np.square(harmonicRMS)))
    return harmonicRMS.tolist()

# NCVD Data Generator
class ejNCVdataGenerator():
    def __init__(self, Vamp_rms, VthdMax, f_fund, t_start, phase, harmonicOrder = 4, sampleLen = 0.05, sampleFreq = 16000, C1 = 18.0E-12, C2 = 19.0E-12, enableVinNoise = True):
        
        self.enableVinNoise = enableVinNoise
        
        self.Vamp_rms, self.VthdMax, self.f_fund, self.phase = Vamp_rms, VthdMax, f_fund, phase
        self.harmoAmpRms = harmonicAmpGen(self.Vamp_rms, self.VthdMax, harmonicOrder)
        self.VacComponent = []
        self.VinComponent = []
        
        self.sampleLen = sampleLen   #numWavePeriod
        self.sampleFreq = sampleFreq #numSample
        self.numSample = int(sampleLen*sampleFreq)
        
        self.start = t_start
        self.time = np.linspace(self.start, self.start+self.sampleLen, num=self.numSample, dtype=float)
        
        self.C1 = C1#18.0E-12 * random.uniform(0.3, 3)
        self.C2 = C2#19.0E-12 * random.uniform(0.3, 3)

        self.VacData = np.zeros(shape = [self.numSample,])
        self.VinData = np.zeros(shape = [self.numSample,])
    
    def checkInitialValue(self):
        print(f"Vrms:\t{self.Vamp_rms}")
        print("Max THD%:\t{:.2%}".format(self.VthdMax))
        print("f1:\t{}".format(self.f_fund))

        print("Actual THD%:\t{:.2%}".format(math.sqrt(np.sum(np.square(np.array(self.harmoAmpRms[1:])))) / self.Vamp_rms))
        print(self.harmoAmpRms[1:])
        print(f"C1: {self.C1}, C2: {self.C2}")
        print(f"Ratio: {self.C1/18.0E-12:.2f}, {self.C2/19.0E-12:.2f}")
        print()
        

    def generateVacComponent(self):
        
        self.harmoAmpRms.insert(0, self.Vamp_rms)
        
        for i, ampRms in enumerate(self.harmoAmpRms):
            phase = self.phase[i]
            tempComponent = sineWave(ampRms, self.f_fund, 2*i+1, phase)
            self.VacComponent.append(tempComponent)
            
    def transVac2Vin(self):
        # Rin: 4  Mohm
        # Cin: 39 pF
        # C1:  18 pF
        # C2:  19 pF
        #self.C1 = 18.0E-12 * random.uniform(0.3, 3)
        #self.C2 = 19.0E-12 * random.uniform(0.3, 3)
        for Vac in self.VacComponent:
            tempComponent = trans2Vin(Vac, 4.0E6, 39.0E-12, self.C1, self.C2)
            self.VinComponent.append(tempComponent)
        
    def addGaussianNoise2Output(self):
        noiseAmp = np.random.uniform(0.1, 0.3, 1)
        mean = 0
        std = 1 

        noise = noiseAmp * np.random.normal(mean, std, size=self.numSample)
        self.VinData += noise
    
    def combinateAllData(self):
        # Truth Vac
        for Vac in self.VacComponent:
            self.VacData += sineValue2Data(Vac, self.time)

        # Truth Vin
        for Vin in self.VinComponent:
            self.VinData += sineValue2Data(Vin, self.time)
    
    def generateAllData(self):
        
        self.generateVacComponent()
        self.transVac2Vin()
        
        self.combinateAllData()
        if self.enableVinNoise:
            self.addGaussianNoise2Output()

    def plotVacAndVinData(self, LineWD=2.0):
        LineWD = LineWD  # Plot Line Width
        fig=plt.figure(figsize=(12,8), dpi= 300, facecolor='w', edgecolor='k')
        plt.plot(self.time, self.VacData,'b-', label='Vac',linewidth=LineWD)
        plt.plot(self.time, 10*self.VinData,'g-', label='Vin',linewidth=LineWD)
        plt.legend()
        plt.show()