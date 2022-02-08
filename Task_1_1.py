import random
import librosa
import librosa.display
import os
import csv
import soundfile as sf
from pydub import AudioSegment
import random
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
#IPython.display as ipd
#ipd.Audio(audio_data)
import pandas as pd
import pywt
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.metrics import mean_squared_error
def get_features(file_name):
    X, sample_rate = librosa.load(file_name, sr=16000)

    stft = np.abs(librosa.stft(X))
    result = np.array([])

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma))
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, contrast))
    cA, cD = pywt.dwt(X, 'db1')
    result = np.hstack((result, [cA.mean(), cD.mean()]))
    return result
i=[]
i_g=[]
tsv_file = open("uk/validated.tsv",encoding='utf-8')
read_tsv = csv.reader(tsv_file, delimiter="\t")
for row in read_tsv:
  #print(row)
    if len(row)>1:
        #print(row[1])
        if(row[6]=="female"):
            i_g.append(1)
            i.append(row[1][16:-4])
        if(row[6]=="male"):
            i_g.append(0)
            i.append(row[1][16:-4])

x, sr = librosa.load('common_voice_uk_25803029.wav',
                     sr = 16000,
                     mono=True,
                     offset=0.0,
                     duration=2.0)
len_i=14000
N_v=1000
d=[]
for ii in range(N_v):
    rand_num=random.randint(0, len_i)
    while(rand_num in set(d)):
        rand_num=random.randint(0, len_i)
    d.append(rand_num)
f = sf.SoundFile("uk/in_wav/common_voice_uk_"+i[ii]+".wav")
ind_min=len(f) / f.samplerate
for ii in range(1,N_v):
    f = sf.SoundFile("uk/in_wav/common_voice_uk_"+i[d[ii]]+".wav")
    ind_min=min(ind_min,len(f) / f.samplerate)
print(ind_min)
    
    
'''for ii in range(2):
    print("uk/in_wav/common_voice_uk_"+i[ii]+".wav")
    x, sr = librosa.load("uk/in_wav/common_voice_uk_"+i[ii]+".wav",
                     sr = 16000,
                     mono=True,
                     offset=0.0,
                     duration=ind_min)
    S_full, phase = librosa.magphase(librosa.stft(x))
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max),
                               y_axis='log', x_axis='time', sr=sr, ax=ax)
    fig.colorbar(img, ax=ax)
    #show()
    plt.grid()
    plt.show()
    librosa.display.waveplot(x, sr=sr)
    X = librosa.stft(x)
    plt.figure(figsize=(14, 5))
    Xdb = librosa.amplitude_to_db(abs(X)) 
    librosa.display.specshow(Xdb, sr=sr, x_axis='время', y_axis ='hz')
    plt.colorbar()
    librosa.display.specshow(Xdb, sr=sr, x_axis='время', y_axis='log')
    plt.colorbar()'''
column_names = []
array = []
pathes=[]
for ii in range(N_v):
    features = get_features("uk/in_wav/common_voice_uk_"+i[d[ii]]+".wav")
    array.append(features)
    pathes.append("uk/in_wav/common_voice_uk_"+i[d[ii]]+".wav")
    print(ii)
for j in range(147):
    if j < 12: column_names.append('chroma_' + str(j))
    elif j>=12  and j<140: column_names.append('mel_' + str(j-12))
    else: column_names.append('contrast_' + str(j-140))
column_names.append('wavelet_cA')
column_names.append('wavelet_cD')
dataframe = pd.DataFrame(array, columns=np.array(column_names))
dataframe['path'] = pathes
pathes_g=[]
for ii in range(N_v):
    pathes_g.append(i_g[d[ii]])

merged_dataframe = pd.merge(dataframe, pd.DataFrame({'path':pathes,
                                                     'gender':pathes_g}), on='path')
X_train, X_test, y_train, y_test = train_test_split(merged_dataframe.drop(columns = ['path', 'gender']),
                                                    merged_dataframe.gender, test_size=0.2, random_state=1)
parametrs = {'n_estimators':range(5,41,5), 'max_depth':range(1,9,2),
    'min_samples_leaf':range(1,8), 'min_samples_split':range(2,4)}
voice_rf = RandomForestClassifier(random_state=0)
search = GridSearchCV(voice_rf, parametrs, n_jobs=-1, cv=3)
search.fit(X_train,y_train)
best_voice_forest = search.best_estimator_

predictions = best_voice_forest.predict(X_test)
(predictions == y_test.to_numpy()).sum()
plot_confusion_matrix(best_voice_forest, X_test, y_test)
plt.show()
    
    
    
    
