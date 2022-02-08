import pandas as pd
import numpy as np
import librosa as lr
from scipy.signal.signaltools import wiener
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import pywt

# Search for audio data
files = lr.util.find_files(r'C:\Users\markm\Desktop\project\2022-knu-bioacoustics\bones_data')
files = np.asarray(files)

# Initialize audio data (Maria - 118 audio, me - 120 audio)
output_list = [0]*118 + [1]*120
targetframe = pd.DataFrame({'path':files, 'output':output_list})

# Create a class that help us to extract features and add them into pandas dataframe
class AudioFeatures:
    def __init__(self, files, path_column, sample_rate=16000):
        self.files = files
        self.path_column = path_column
        self.sample_rate = sample_rate
        self.audios = []
        self.features = []

    def extract_all_features(self):
        self.load()
        self.energy_normalize()
        self.get_features()
        return self.create_features_df()

    def load(self):
        for path in range(len(self.path_column)):
            audio, sample_rate = lr.load(self.path_column.values[path], sr=self.sample_rate)
            self.audios.append(audio)

    def energy_normalize(self):
        for i, audio in enumerate(self.audios):
            w_audio = wiener(audio)
            current_energy = np.sum(w_audio ** 2)
            self.audios[i] = w_audio / np.sqrt(current_energy)

    def get_features(self):
        for audio in self.audios:
            stft = np.abs(lr.stft(audio))
            result = np.array([])
            chroma = np.mean(lr.feature.chroma_stft(S=stft, sr=self.sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
            mel = np.mean(lr.feature.melspectrogram(audio, sr=self.sample_rate).T, axis=0)
            result = np.hstack((result, mel))
            contrast = np.mean(lr.feature.spectral_contrast(S=stft, sr=self.sample_rate).T, axis=0)
            result = np.hstack((result, contrast))
            cA, cD = pywt.dwt(audio, 'db1')
            result = np.hstack((result, [cA.mean(), cD.mean()]))
            self.features.append(result)

    def create_features_df(self):
        column_names = []
        for j in range(147):
            if j < 12:
                column_names.append('chroma_' + str(j))
            elif j >= 12 and j < 140:
                column_names.append('mel_' + str(j - 12))
            else:
                column_names.append('contrast_' + str(j - 140))
        column_names.append('wavelet_cA')
        column_names.append('wavelet_cD')
        featuresframe = pd.DataFrame(self.features, columns=np.array(column_names))
        featuresframe['path'] = files
        return featuresframe

AF = AudioFeatures(files, targetframe.path)
featuresframe = AF.extract_all_features()

# Make a dataframe: features + output
merged_dataframe = pd.merge(featuresframe, targetframe, on='path')

# Modelling and cross-validation
X_train, X_test, y_train, y_test = train_test_split(merged_dataframe.drop(columns = ['path', 'output']),
                                                    merged_dataframe.output, test_size=0.3, random_state=3)
parametrs = {'n_estimators':range(5,41,5), 'max_depth':range(1,7),
    'min_samples_leaf':range(1,8), 'min_samples_split':range(2,4)}
voice_rf = RandomForestClassifier(random_state=0)
search = GridSearchCV(voice_rf, parametrs, n_jobs=-1, cv=5)
search.fit(X_train,y_train)
best_voice_forest = search.best_estimator_

predictions = best_voice_forest.predict(X_test)
scores = cross_val_score(best_voice_forest, merged_dataframe.drop(columns = ['path', 'output']),
                         merged_dataframe.output, cv=5)
print(f'Accuracy: {(predictions == y_test.to_numpy()).sum()}/{len(predictions)}')
print('Cross-validation score: ', scores.mean())
print(confusion_matrix(y_test, predictions))
print("Mean squared error: %.2f"
    % mean_squared_error(y_test, predictions))