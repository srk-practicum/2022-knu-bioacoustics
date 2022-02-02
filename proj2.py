import os
from glob import glob
from typing import Any
import sklearn
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

x, sr = librosa.load('common_voice_uk_25803029.wav', sr = 16000, mono=True, offset=0.0, duration=3.0)
# показать спектр
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)

# показать спектрограмму
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()

# Построение спектрального центроида
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
#print(spectral_centroids)
# Вычисление временной переменной для визуализации
plt.figure(figsize=(12, 4))
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames, sr = 16000)

# Нормализация спектрального центроида для визуализации
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
# Построение спектрального центроида вместе с формой волны
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='b')

# Построение спектрального спада
spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
plt.figure(figsize=(12, 4))
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')

# Построение спектральной ширины
spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr)[0]
spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=3)[0]
spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=4)[0]
plt.figure(figsize=(15, 9))

librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_bandwidth_2), color='r')
plt.plot(t, normalize(spectral_bandwidth_3), color='g')
plt.plot(t, normalize(spectral_bandwidth_4), color='y')
plt.legend(('p = 2', 'p = 3', 'p = 4'))

plt.show()

''' time = np.arange(0, len(audio)) /sr
print(time)
print(audio, sf)
plt.figure(figsize=(14, 5))
librosa.display.waveplot(audio, sr=sf)
fig, ax = plt.subplots()
ax.plot(time, audio)
ax.set(xlabel = 'Time', ylabel = 'Sound amplitud')
plt.show()
for file in range(0, len(audio_files), 1):
    audio, sf = librosa.load(audio_files[file], sr=16000, mono=True, offset=0.0, duration=3.0)
    time = np.arange(0, len(audio)) / sf
    fig, ax = plt.subplots()
    ax.plot(time, audio)

    spectral_centroids = librosa.feature.spectral_centroid(audio, sr=sf)[0]
    #    print(spectral_centroids.shape)

    frames = range(len(spectral_centroids))
    print(frames)
    print(len(spectral_centroids))
    t = librosa.frames_to_time(frames)


    #    print(t)
    # Нормализация спектрального центроида для визуализации
    def normalize(x, axis=0):
        return sklearn.preprocessing.minmax_scale(x, axis=axis)


    # Построение спектрального центроида вместе с формой волны
    librosa.display.waveplot(audio, sr=sf, alpha=0.4)
    plt.plot(t, normalize(spectral_centroids), color='b')
    ax.set(xlabel='Time' + y[file], ylabel='Sound amplitud')

    spectral_rolloff: object = librosa.feature.spectral_rolloff(audio + 0.01, sr=sf)[0]
    plt.figure(figsize=(12, 4))

    librosa.display.waveplot(audio, sr=sf, alpha=0.4)
    plt.plot(t, normalize(spectral_rolloff), color='r')
    print(y[file])
    ax.set(xlabel='Time' + y[file], ylabel='Sound amplitud')

    plt.show()
    '''
