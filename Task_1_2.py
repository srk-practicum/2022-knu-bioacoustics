import csv
import pandas as pd
import chardet
from scipy import signal
from scipy.fftpack import fft, fftshift
import matplotlib.pyplot as plt
import pywt
import numpy as np
import scipy as sp
import librosa
from scipy.signal.signaltools import wiener
def energy_normalize(y):
    y1 = wiener(y)
    current_energy = np.sum(y1 ** 2)
    return y1 / np.sqrt(current_energy)
def split_ay(file):
    count=librosa.get_duration(filename=file)//2
    d=dict()
    
    for i in range(int(count)):
        y=energy_normalize(librosa.load(file, sr=8000,offset=i*2+1,duration=2)[0])
        d["sound"+str(i)]=y
    return d


def get_features(y):
    stft = np.abs(librosa.stft(y))
    
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=16000).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y, sr=16000).T,axis=0)
    cA, cD = pywt.dwt(y, 'coif3')
    return [*chroma, *mel, cA.mean(), cD.mean()]

def get_controls_feature(d):
    res_cont=[]
    for i in d.keys():
        res_cont.append(get_features(d[i]))
    return res_cont

        
                        

feat=[]

feat=get_controls_feature(split_ay("data/u1_1.wav"))
feat=[*feat ,*get_controls_feature(split_ay("data/u1_2.wav"))]
y_binary=[1]*len(feat)
feat=[*feat,*get_controls_feature(split_ay("data/u2_2.wav"))]
y_binary=[1]*len(y_binary)+[0]*(len(feat)-len(y_binary))
clms = ['chroma'+' {}'.format(i) for i in range(1,13)] + ['mel'+' {}'.format(i) for i in range(1,129)] + ['cA', 'cD']
merged_df= pd.DataFrame(feat, columns=clms)
X = merged_df.values
df = pd.DataFrame({'target':y_binary})
y_binary = df.target.values

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=2)

for train_index, test_index in sss.split(X, y_binary):
    #print(train_index, test_index)
    X_train=X[train_index.astype(int)]
    X_test = X[test_index.astype(int)]
    y_train, y_test = y_binary[train_index.astype(int)], y_binary[test_index.astype(int)]
    print("Random Forest Classifier\n")
    clf=RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    FRR, FAR = 0, 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_test[i] == 0:
            FAR+=1
        if y_pred[i] == 0 and y_test[i] == 1:
            FRR+=1
            
    print('FAR = {} FRR = {}'.format( FAR,FRR))
    print('all = {}\n'.format(len(y_pred)-FAR-FRR))

    print("SVC")
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    FRR, FAR = 0, 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_test[i] == 0:
            FAR+=1
        if y_pred[i] == 0 and y_test[i] == 1:
            FRR+=1
    print('FAR = {} FRR = {}'.format( FAR,FRR))
    print('all = {}\n'.format(len(y_pred)-FAR-FRR))
    print("Decision Tree Classifier")
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    FRR, FAR = 0, 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_test[i] == 0:
            FAR+=1
        if y_pred[i] == 0 and y_test[i] == 1:
            FRR+=1
    print('FAR = {} FRR = {}'.format( FAR,FRR))
    print('all = {}\n'.format(len(y_pred)-FAR-FRR))
    
    

'''

Результат работы:

Random Forest Classifier

FAR = 8 FRR = 5
all = 67

SVC
FAR = 5 FRR = 3
all = 72

Decision Tree Classifier
FAR = 12 FRR = 8
all = 60

Random Forest Classifier

FAR = 4 FRR = 2
all = 74

SVC
FAR = 6 FRR = 2
all = 72

Decision Tree Classifier
FAR = 4 FRR = 7
all = 69

Random Forest Classifier

FAR = 5 FRR = 0
all = 75

SVC
FAR = 3 FRR = 2
all = 75

Decision Tree Classifier
FAR = 7 FRR = 5
all = 68

Random Forest Classifier

FAR = 4 FRR = 1
all = 75

SVC
FAR = 4 FRR = 2
all = 74

Decision Tree Classifier
FAR = 6 FRR = 7
all = 67

Random Forest Classifier

FAR = 7 FRR = 2
all = 71

SVC
FAR = 4 FRR = 1
all = 75

Decision Tree Classifier
FAR = 10 FRR = 5
all = 65

Random Forest Classifier

FAR = 8 FRR = 3
all = 69

SVC
FAR = 3 FRR = 5
all = 72

Decision Tree Classifier
FAR = 8 FRR = 7
all = 65

Random Forest Classifier

FAR = 4 FRR = 1
all = 75

SVC
FAR = 4 FRR = 1
all = 75

Decision Tree Classifier
FAR = 6 FRR = 9
all = 65

Random Forest Classifier

FAR = 3 FRR = 3
all = 74

SVC
FAR = 3 FRR = 4
all = 73

Decision Tree Classifier
FAR = 5 FRR = 7
all = 68

Random Forest Classifier

FAR = 2 FRR = 4
all = 74

SVC
FAR = 4 FRR = 4
all = 72

Decision Tree Classifier
FAR = 5 FRR = 5
all = 70

Random Forest Classifier

FAR = 12 FRR = 6
all = 62

SVC
FAR = 8 FRR = 5
all = 67

Decision Tree Classifier
FAR = 8 FRR = 8
all = 64'''
