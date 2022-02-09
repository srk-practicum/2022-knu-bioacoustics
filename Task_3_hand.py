import librosa
import pandas as pd
import random 
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import scipy as sp
import pywt




files = librosa.util.find_files('E:\data_voices\hand')
df = pd.DataFrame({'path': np.asarray(files), 'target': [0]*118+[1]*31+[0]*121})
def audio_filters(file):
    y, sr = librosa.load(file, sr = 16000)
    #Median filter
    y1 = sp.signal.medfilt(y,9)
    #Wevelets ??
    y1, cD2 = pywt.dwt(y, 'db2')
    cA3, cD3 = pywt.dwt(y, 'db3')
    #Wiener filter
    y1=sp.signal.wiener(y1)
    return y1





def extract_features(y):
    stft = np.abs(librosa.stft(y))
    
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=16000).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y, sr=16000).T,axis=0)
    cA, cD = pywt.dwt(y, 'db1')
    return [*chroma, *mel, cA.mean(), cD.mean()]





features_list = []
for file in df['path'].values:
    file = audio_filters(file)
    features_list.append(extract_features(file))




clms = ['chroma'+' {}'.format(i) for i in range(1,13)] + ['mel'+' {}'.format(i) for i in range(1,129)] + ['cA', 'cD']
df_features = pd.DataFrame(features_list, columns=clms)
df_features['path'] = df['path'].values
merged_df = df.merge(df_features)




merged_df




from sklearn.model_selection import StratifiedShuffleSplit
X = merged_df.drop(columns = ['path','target'], axis=1).values
y = df.target.values
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=1)
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier()

for train_index, test_index in sss.split(X, y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    FRR, FAR = 0, 0
    #print(len(y_pred))
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_test[i] == 0:
            FAR+=1
        if y_pred[i] == 0 and y_test[i] == 1:
            FRR+=1
            
    print('FAR = ', FAR/len(y_pred))
    print('FRR = ', FRR/len(y_pred))
    print((y_pred == y_test).sum()/len(y_test))
    print('\n')






