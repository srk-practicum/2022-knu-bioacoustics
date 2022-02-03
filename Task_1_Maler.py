import pandas as pd
import numpy as np
import librosa as lr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from pydub import AudioSegment
import pywt

def sex(x):
    if x == 'male': return 1
    else: return 0
def get_features(file_name):
    X, sample_rate = lr.load(file_name, sr=16000)
    stft = np.abs(lr.stft(X))
    result = np.array([])
    chroma = np.mean(lr.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma))
    mel = np.mean(lr.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
    contrast = np.mean(lr.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, contrast))
    cA, cD = pywt.dwt(X, 'db1')
    result = np.hstack((result, [cA.mean(), cD.mean()]))
    return result

common_voices_df = (pd.read_csv(r'E:\common_voices\uk\validated.tsv', sep='\t')[['path', 'gender']]).dropna()
common_voices_sample = common_voices_df.sample(n=3000, random_state=1)
common_voices_sample.gender = common_voices_sample.gender.apply(sex)
for file in common_voices_sample['path'].values:
    sound = AudioSegment.from_mp3(r'E:\common_voices\uk\clips\\' + file)
    sound.export(r'E:\common_voices\uk\clips_wav\\' + file, format='wav')
files = lr.util.find_files(r'E:\common_voices\uk\clips_wav')
files = np.asarray(files)
sample_pathes = []
for file in files:
    if (((file.split('\\')[4]).split('.'))[0] + '.mp3') in common_voices_sample.path.values:
        sample_pathes.append(file)
array = []
for k in range(len(sample_pathes)):
    features = get_features(sample_pathes[k])
    array.append(features)
column_names = []
for j in range(147):
    if j < 12: column_names.append('chroma_' + str(j))
    elif j>=12  and j<140: column_names.append('mel_' + str(j-12))
    else: column_names.append('contrast_' + str(j-140))
column_names.append('wavelet_cA')
column_names.append('wavelet_cD')
dataframe = pd.DataFrame(array, columns=np.array(column_names))
dataframe['path'] = sample_pathes
merged_dataframe = pd.merge(dataframe, common_voices_sample, on='path')

X_train, X_test, y_train, y_test = train_test_split(merged_dataframe.drop(columns = ['path', 'gender']),
                                                    merged_dataframe.gender, test_size=0.2, random_state=1)
parametrs = {'n_estimators':range(5,41,5), 'max_depth':range(1,9,2),
    'min_samples_leaf':range(1,8), 'min_samples_split':range(2,4)}
voice_rf = RandomForestClassifier(random_state=0)
search = GridSearchCV(voice_rf, parametrs, n_jobs=-1, cv=3)
search.fit(X_train,y_train)
best_voice_forest = search.best_estimator_
predictions = best_voice_forest.predict(X_test)

print(confusion_matrix(y_test, predictions))
print("Mean squared error: %.2f"
    % mean_squared_error(y_test, predictions))