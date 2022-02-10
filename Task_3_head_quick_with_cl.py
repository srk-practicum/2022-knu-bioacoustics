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

def get_features(y):
    stft = np.abs(librosa.stft(y))
    
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=16000).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y, sr=16000).T,axis=0)
    cA, cD = pywt.dwt(y, 'db2')
    return [*chroma, *mel, cA.mean(), cD.mean()]

def get_controls_feature(path_):
    files = librosa.util.find_files(path_)
    print(files)
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

start_time = time.time()
merged_df = pd.read_csv('data_head.csv')
y = merged_df.target.values

X = merged_df.drop(columns = ['path','target','Unnamed: 0'], axis=1).values


print("--- %s seconds ---" % (time.time() - start_time))


path_='E:\Head'
X_for_test=get_controls_feature(path_)
#RandomForestClassifier
start_time = time.time()
clf=RandomForestClassifier()
clf.fit(X, y)
y_pred = clf.predict(X_for_test)
print("--- %s seconds ---" % (time.time() - start_time))
print(y_pred)
#GaussianNB
start_time = time.time()
clf = GaussianNB()
clf.fit(X, y)
y_pred = clf.predict(X_for_test)
print("--- %s seconds ---" % (time.time() - start_time))
print(y_pred)
#MLPClassifier
start_time = time.time()
clf = MLPClassifier(random_state=1, max_iter=300).fit(X, y)
y_pred = clf.predict(X_for_test)
print("--- %s seconds ---" % (time.time() - start_time))
print(y_pred)
#DecisionTreeClassifier
start_time = time.time()
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X, y)
y_pred = clf.predict(X_for_test)
print("--- %s seconds ---" % (time.time() - start_time))
print(y_pred)


f1=plt.subplot(3,2,1)
f1.set_title('Using Affinity Propagation')
model = AffinityPropagation(damping=0.9)
model.fit(X)
yhat = model.predict(X)
clusters = np.unique(yhat)
for cluster in clusters:
	row_ix = np.where(yhat == cluster)
	plt.scatter(X[row_ix, 0], X[row_ix, 1])

f2=plt.subplot(3,2,2)
f2.set_title('Using DBSCAN Clustering')
model = DBSCAN(eps=0.30, min_samples=4)
yhat = model.fit_predict(X)
clusters = np.unique(yhat)
for cluster in clusters:
	row_ix = np.where(yhat == cluster)
	plt.scatter(X[row_ix, 0], X[row_ix, 1])

f3=plt.subplot(3,2,3)
f3.set_title('Using K-Means Clustering')
model = KMeans(n_clusters=4)
model.fit(X)
yhat = model.predict(X)
clusters = np.unique(yhat)
for cluster in clusters:
	row_ix = np.where(yhat == cluster)
	plt.scatter(X[row_ix, 0], X[row_ix, 1])
labels = model.labels_
print(metrics.silhouette_score(X, labels, metric='euclidean'))
print(davies_bouldin_score(X, labels))

f4=plt.subplot(3,2,4)
f4.set_title('Using OPTICS Clustering')
model = OPTICS(eps=0.8, min_samples=4)
yhat = model.fit_predict(X)
clusters = np.unique(yhat)
for cluster in clusters:
	row_ix = np.where(yhat == cluster)
	plt.scatter(X[row_ix, 0], X[row_ix, 1])

f5=plt.subplot(3,2,5)
f5.set_title('Using Spectra Clustering')
model = KMeans(n_clusters=4)
yhat = model.fit_predict(X)
clusters = np.unique(yhat)
for cluster in clusters:
	row_ix = np.where(yhat == cluster)
	plt.scatter(X[row_ix, 0], X[row_ix, 1])


f6=plt.subplot(3,2,6)
f6.set_title('Using Gaussian Mixture Clustering')
model = GaussianMixture(n_components=4)
model.fit(X)
yhat = model.predict(X)
clusters = np.unique(yhat)
for cluster in clusters:
	row_ix = np.where(yhat == cluster)
	plt.scatter(X[row_ix, 0], X[row_ix, 1])
	print(X[row_ix, 0], X[row_ix, 1])
print(":)")
plt.show()



