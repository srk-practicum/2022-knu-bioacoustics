import pywt
import librosa
import soundfile
import matplotlib.pyplot as plt
from scipy.signal.signaltools import wiener
from pydub import AudioSegment
from pydub import effects
import scipy

def normalise_1(file_path,file_path2):
    
    sound_before = AudioSegment.from_file(file_path)
    sound_after = effects.normalize(sound_before)
    sound_after.export(file_path2, format="wav")
def normalise_2(y):
        max_en=np.sum(y ** 2)
        return y/sqrt(max_en)
def median_filt(y):
    return wiener((scipy.signal.medfilt(y,200)))
