import pywt
import librosa
import soundfile
import matplotlib.pyplot as plt
from scipy.signal.signaltools import wiener
from pydub import AudioSegment
from pydub import effects
import scipy
import numpy as np
import math
def normalise_1(file_path,file_path2):
    
    sound_before = AudioSegment.from_file(file_path)
    sound_after = effects.normalize(sound_before)
    sound_after.export(file_path2, format="wav")
def normalise_2(y):
        max_en=np.sum(y ** 2)
        return y/math.sqrt(max_en)
def median_filt(y):
    return wiener((scipy.signal.medfilt(y,7)))
files = librosa.util.find_files('E:\data_voices\hand')
for i in files:
    normalise_1(i,'E:\data_voices\hand_after\\'+i[20::])
    y,sr=librosa.load('E:\data_voices\hand_after\\'+i[20::], sr = 16000)
    y=median_filt(normalise_2(y))
    soundfile.write('E:\data_voices\hand_after\\'+i[20::],y, sr)

    
    
    
        
    


