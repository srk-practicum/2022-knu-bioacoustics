import librosa
import pandas as pd
import random 
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import scipy as sp
import pywt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import time

files = librosa.util.find_files('E:\data_voices\head')
df = pd.DataFrame({'path': np.asarray(files), 'target': [0]*120+[1]*71+[0]*45})

def get_features(y):
    stft = np.abs(librosa.stft(y))
    
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=16000).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y, sr=16000).T,axis=0)
    cA, cD = pywt.dwt(y, 'db2')
    return [*chroma, *mel, cA.mean(), cD.mean()]

def get_controls_feature(path_):
    files = librosa.util.find_files(path_)
    df = pd.DataFrame({'path': np.asarray(files)})
    res_cont=[]
    for file in df['path'].values:
        res_cont.append(get_features(librosa.load(file, sr = 16000)[0]))
    clms_cont = ['chroma'+' {}'.format(i) for i in range(1,13)] + ['mel'+' {}'.format(i) for i in range(1,129)] + ['cA', 'cD']
    features_cont = pd.DataFrame(res_cont, columns=clms_cont)
    features_cont['path'] = df['path'].values
    merged_df_cont = df.merge(features_cont)
    #print(len(merged_df_cont),len(merged_df_cont[0]))
    X = merged_df_cont.drop(columns = ['path'], axis=1).values
    return X



features_list = []
for file in df['path'].values:
    features_list.append(get_features(librosa.load(file, sr = 16000)[0]))




clms = ['chroma'+' {}'.format(i) for i in range(1,13)] + ['mel'+' {}'.format(i) for i in range(1,129)] + ['cA', 'cD']
df_features = pd.DataFrame(features_list, columns=clms)
df_features['path'] = df['path'].values
merged_df = df.merge(df_features)


X = merged_df.drop(columns = ['path','target'], axis=1).values
y = df.target.values

path_='E:\data_voices\head_test'
X_for_test=get_controls_feature(path_)
start_time = time.time()
clf=RandomForestClassifier()



clf.fit(X, y)

y_pred = clf.predict(X_for_test)
print("--- %s seconds ---" % (time.time() - start_time))
print(y_pred)
'''FRR, FAR = 0, 0
for i in range(len(y_pred)):
    if y_pred[i] == 1 and y[i] == 0:
        FAR+=1
    if y_pred[i] == 0 and y[i] == 1:
        FRR+=1
            
print('FAR = ', FAR/len(y_pred))
print('FRR = ', FRR/len(y_pred))'''








