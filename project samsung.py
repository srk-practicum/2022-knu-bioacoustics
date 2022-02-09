import pandas
import numpy
import librosa
from scipy.signal import wiener
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import pywt
import os

# Search for audio data
files = librosa.util.find_files(r'E:/samsung/bones_data')
files = numpy.asarray(files)

# Initialize audio data (Maria - 118 audio, Mark - 120 audio, Sasha - 102 audio, Stas - 105)
output_list = [0]*118 + [0]*120 + [0]*102 + [1]*105
targetframe = pandas.DataFrame({'path':files, 'output':output_list})

def extract_all_features():
#        audio files loading
    for filename in os.listdir(r'E:\samsung\bones_data'):
        sound = f'E:/samsung/bones_data/{filename}'
        audio, sample_rate = librosa.load(sound, sr=16000)
        audios.append(audio)
#        filter
    for i, audio in enumerate(audios):
        w_audio = wiener(audio)
        current_energy = numpy.sum(w_audio ** 2)
        audios[i] = w_audio / numpy.sqrt(current_energy)
#        features selection
    for audio in audios:
        stft = numpy.abs(lr.stft(audio))
        result = numpy.array([])
        chroma = numpy.mean(librosa.feature.chroma_stft(S=stft, sr=16000).T, axis=0)
        result = numpy.hstack((result, chroma))
        mel = numpy.mean(librosa.feature.melspectrogram(audio, sr=16000).T, axis=0)
        result = numpy.hstack((result, mel))
        contrast = numpy.mean(librosa.feature.spectral_contrast(S=stft, sr=16000).T, axis=0)
        result = numpy.hstack((result, contrast))
        cA, cD = pywt.dwt(audio, 'db1')
        result = numpy.hstack((result, [cA.mean(), cD.mean()]))
        features.append(result)

    column_names = []
    for j in range(147):
        if j < 12:
            column_names.append('chroma_' + str(j))
        elif (j >= 12) and (j < 140):
            column_names.append('mel_' + str(j - 12))
        else:
            column_names.append('contrast_' + str(j - 140))
    column_names.append('wavelet_cA')
    column_names.append('wavelet_cD')
    featuresframe = pandas.DataFrame(features, columns=numpy.array(column_names))
    return featuresframe


features_frame = extract_all_features

# Make a dataframe: features + output
merged_dataframe = pandas.merge(features_frame, targetframe, on='path')

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