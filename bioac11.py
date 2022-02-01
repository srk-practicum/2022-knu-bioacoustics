import os
import random
import librosa
import pandas as pd
import numpy as np
import soundfile

def get_features(file_name, **boolean_kwargs):
    chroma_b = boolean_kwargs.get("chroma")
    mel_b = boolean_kwargs.get("mel")
    tonnetz_b = boolean_kwargs.get("tonnetz")
    contrast_b = boolean_kwargs.get("contrast")
    mfcc_b = boolean_kwargs.get("mfcc")

    X, sample_rate = librosa.load(file_name, sr=16000)
    e = (X**2).sum()
    X = X/e**(1/2)
    stft = np.abs(librosa.stft(X))
    result = np.array([])

    if chroma_b:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
    if mel_b:
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
    if tonnetz_b:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
        result = np.hstack((result, tonnetz))
    if contrast_b:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, contrast))
    if mfcc_b:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    return result

#dir_list = os.listdir('clips/')
characteristics = pd.read_csv('uk/validated.tsv', sep='\t')
dir_list = characteristics['path'].values
random.seed(3)
files = []
N=100
for i in range(N):
    k = random.randint(0, len(dir_list)-1)
    files.append(dir_list[k])
print(files)

choised_audios = []
for i in range(N):
    y, sr = librosa.load('clips/'+files[i],sr=16000)
    if i == 0:
        min_index = 0
        min_dur = librosa.get_duration(y, sr)
    elif librosa.get_duration(y,sr) < min_dur:
        min_index = i
        min_dur = librosa.get_duration(y, sr)
    choised_audios.append([y, sr])
print('min_dur: ',min_dur)

for i in range(N):
    sr = 16000
    y = choised_audios[i][0]
    print(y, sr)
    y = y[:int(sr*min_dur)]
    choised_audios[i][0] = y
    choised_audios[i][1] = sr

    soundfile.write('clips/new___'+files[i][:-3]+'wav', y, sr)

array = []
for k in range(len(files)):
    features = get_features('clips/new___'+files[k][:-3]+'wav', chroma=True, mel=True)
    array.append(features)

dataframe = pd.DataFrame({})

def age_to_number(age):
    if age=='teens':
        return 1
    elif age=='twenties':
        return 2
    elif age=='thirties':
        return 3
    else:
        return 4
def sex(x):
    if x != 'female':
        return 0
    else:
        return 1

characteristics['age']=characteristics['age'].apply(age_to_number)
characteristics['gender']=characteristics['gender'].apply(sex)

for i in range(len(files)):
    audio_dict = {}
    audio_dict['path'] = files[i]
    for j in range(len(array[i])):

        if j < 12:
            audio_dict['chroma_' + str(j)] = array[i][j]
        else:
            audio_dict['mel_' + str(j - 12)] = array[i][j]
    result = characteristics[characteristics['path'] == files[i]]
    print(result['gender'].values[0])
    audio_dict['gender'] = result['gender'].values[0]
    dataframe = dataframe.append(audio_dict, ignore_index=True)
X = dataframe[dataframe.columns[1:-1]].values
y = dataframe[dataframe.columns[-1]].values
print('X = ',X.shape)
print('y = ',y)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

clf=RandomForestClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = (y_pred == y_test).sum()/len(y_test)
print(accuracy)


