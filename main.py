import os
from glob import glob
from typing import Any
import sklearn
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

MAIN_DIR = os.path.dirname(__file__)  # Полный путь к каталогу с файлами
# list_dir = os.listdir(MAIN_DIR)
list_dir: list[str] = os.listdir('C:/projects/sounds/')
y = [i for i in list_dir if len(i.split('.'))>1 and i.split('.')[1] == 'mp3']
print(y)
path1 = y[0]
file_dir = 'C:/projects/sounds/'
audio_files: list[Any] = glob(file_dir + '*.wav')

for file in range (0, len(audio_files), 1):
    audio, sr = librosa.load(audio_files[file], sr=16000, mono=True, offset=0.0, duration=3.0)
    # показать спектр
    time = np.arange(0, len(audio)) / sr
    fig, ax = plt.subplots()
    ax.plot(time, audio)

    # показать спектрограмму
    Audio = librosa.stft(audio)
    Audiodb = librosa.amplitude_to_db(abs(Audio))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Audiodb, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()

    # Построение спектрального центроида

    spectral_centroids = librosa.feature.spectral_centroid(audio, sr=sr)[0]

    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames, sr=16000)
    # Нормализация спектрального центроида для визуализации
    def normalize(x, axis=0):
        return sklearn.preprocessing.minmax_scale(x, axis=axis)
    # Построение спектрального центроида вместе с формой волны
    librosa.display.waveplot(audio, sr=sr, alpha=0.4)
    plt.plot(t, normalize(spectral_centroids), color='b')
    ax.set(xlabel='Time' + y[file], ylabel='Sound amplitud')
    # Построение спектрального спада
    spectral_rolloff: object = librosa.feature.spectral_rolloff(audio + 0.01, sr=sr)[0]
    plt.figure(figsize=(12, 4))

    librosa.display.waveplot(audio, sr=sr, alpha=0.4)
    plt.plot(t, normalize(spectral_rolloff), color='r')
    ax.set(xlabel='Time' + y[file], ylabel='Sound amplitud')

    # Построение спектральной ширины
    spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(audio + 0.01, sr=sr)[0]
    spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(audio + 0.01, sr=sr, p=3)[0]
    spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(audio + 0.01, sr=sr, p=4)[0]
    plt.figure(figsize=(15, 9))

    librosa.display.waveplot(audio, sr=sr, alpha=0.4)
    plt.plot(t, normalize(spectral_bandwidth_2), color='r')
    plt.plot(t, normalize(spectral_bandwidth_3), color='g')
    plt.plot(t, normalize(spectral_bandwidth_4), color='y')
    plt.legend(('p = 2', 'p = 3', 'p = 4'))

    plt.show()
