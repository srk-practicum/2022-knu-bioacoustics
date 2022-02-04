import os
import librosa as lr
import scipy.signal as scs
import numpy as np

from dataset.dataloader import Dataloader
from classifier.model import Model

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet_v2 import ResNet50V2

filepath_base = os.path.dirname(os.path.abspath(__file__))
datafolder = os.path.join(filepath_base, './dataset/data')
pklpath = os.path.join(datafolder, 'train.pkl')

dataloader = Dataloader(16000, 1)

cache = dataloader.fit(datafolder, 'train.tsv', 'gender')
dataloader.save(pklpath)

dataloader.load_pkl(pklpath)

knn = KNeighborsClassifier(n_neighbors=7)
rf = RandomForestClassifier()
svm_linear = SVC(kernel='linear')
svm_rbf = SVC()
svm_sigmoid = SVC(kernel='sigmoid')

model = Model(dataloader)

h = model.fit_net(base_model=Xception,
                  features='wavelets',
                  show_model=True,
                  layers_to_freeze=165,
                  epochs=3)


trained = model.train(knn, 0.2, 'stft')
model.estimate(trained['true'], trained['pred'])

trained = model.train(rf, 0.2, 'stft')
model.estimate(trained['true'], trained['pred'])

trained = model.train(svm_linear, 0.2, 'stft')
model.estimate(trained['true'], trained['pred'])

trained = model.train(svm_rbf, 0.2, 'stft')
model.estimate(trained['true'], trained['pred'])

trained = model.train(svm_sigmoid, 0.2, 'stft')
model.estimate(trained['true'], trained['pred'])

