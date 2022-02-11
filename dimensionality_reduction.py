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
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

import time
import csv

from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture

from sklearn.metrics import davies_bouldin_score
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPRegressor

#from keras.models import Model
#from keras.layers import Input, Dense
#from keras import regularizers
from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

merged_df = pd.read_csv('data_head_after.csv')
y = merged_df.target.values

X = merged_df.drop(columns = ['path','target','Unnamed: 0'], axis=1).values
pca = PCA(n_components=len(X[0])//4)
pca.fit(X)
X_new=pca.transform(X)
transformer = KernelPCA(n_components=len(X[0])//4, kernel='sigmoid')
X_new1 = transformer.fit_transform(X)

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=20, test_size=0.5, random_state=1)

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf=RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    FRR, FAR = 0, 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_test[i] == 0:
            FAR+=1
        if y_pred[i] == 0 and y_test[i] == 1:
            FRR+=1
            
    #print('FAR = {} FRR = {}'.format( FAR,FRR))
    print('res = {} %'.format(100*(len(y_pred)-FAR-FRR)/len(y_pred)))
    proc_1=100*(len(y_pred)-FAR-FRR)/len(y_pred)
    X_train, X_test = X_new[train_index], X_new[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf=RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    FRR, FAR = 0, 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_test[i] == 0:
            FAR+=1
        if y_pred[i] == 0 and y_test[i] == 1:
            FRR+=1
            
    #print('PCA: FAR = {} FRR = {}'.format( FAR,FRR))
    print('PCA: res = {} %'.format(100*(len(y_pred)-FAR-FRR)/len(y_pred)))
    print('delta= ',proc_1-100*(len(y_pred)-FAR-FRR)/len(y_pred))

    X_train, X_test = X_new1[train_index], X_new1[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf=RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    FRR, FAR = 0, 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_test[i] == 0:
            FAR+=1
        if y_pred[i] == 0 and y_test[i] == 1:
            FRR+=1
            
    #print('PCA Kernel: FAR = {} FRR = {}'.format( FAR,FRR))
    print('PCA Kernel: res = {} %'.format(100*(len(y_pred)-FAR-FRR)/len(y_pred)))
    print('delta= ',proc_1-100*(len(y_pred)-FAR-FRR)/len(y_pred))


    
    print('\n')


    
