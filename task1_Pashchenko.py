import librosa
import pandas as pd
import random 
import numpy as np
from pydub import AudioSegment
import scipy
import pywt
from sklearn.ensemble import RandomForestClassifier

df_train = pd.read_csv(r"C:\Dataframes\train.tsv", sep='\t')
df_train = df_train.drop(axis=1, columns=['locale', 'up_votes', 'down_votes', 'accent', 'segment', 'age'])
df_train = df_train.dropna()

df_test = pd.read_csv(r"C:\Dataframes\test.tsv",sep='\t')
df_test = df_test.drop(axis=1, columns=['locale', 'up_votes', 'down_votes', 'accent', 'segment', 'age'])
df_test = df_test.dropna()

df_train_sample = df_train.sample(n=1000, random_state=6)
df_test_sample = df_test.sample(n=100, random_state=6)


def gender_func(g):
    if g == 'male':
        return 1
    if g == 'female':
        return 0
    
df_train_sample['gender'] = df_train_sample['gender'].apply(gender_func)
df_test_sample['gender'] = df_test_sample['gender'].apply(gender_func)


for file in df_train_sample['path'].values:
    sound = AudioSegment.from_mp3('C:/clips/'+file)
    sound.export('C:/clips_wav_train/'+file, format='wav')


for file in df_test_sample['path'].values:
    sound = AudioSegment.from_mp3('C:/clips/'+file)
    sound.export('C:/clips_wav_test/'+file, format='wav')



def extract_features(filename):
    
    y, sr = librosa.load(filename, sr = 16000)
    stft = np.abs(librosa.stft(y))
    
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y, sr=sr).T,axis=0)
    
    cA, cD = pywt.dwt(y, 'db1')

    return [*chroma, *mel, cA.mean(), cD.mean()]


train_feachers_list=[]
for file in df_train_sample['path'].values:
    file = 'C:/clips_wav_train/'+file
    train_feachers_list.append(extract_features(file))


clms = ['chroma'+' {}'.format(i) for i in range(1,13)] + ['mel'+' {}'.format(i) for i in range(1,129)] + ['cA', 'cD']
train_df_features = pd.DataFrame(train_feachers_list, columns=clms)
train_df_features['path'] = df_train_sample['path'].values
train_merged_df = df_train_sample.merge(train_df_features)


test_feachers_list=[]
for file in df_test_sample['path'].values:
    file = 'C:/clips_wav_test/'+file
    test_feachers_list.append(extract_features(file))


test_df_features = pd.DataFrame(test_feachers_list, columns=clms)
test_df_features['path'] = df_test_sample['path'].values
test_merged_df = df_test_sample.merge(test_df_features)


clf=RandomForestClassifier()


X_train = train_merged_df.drop(columns = ['path', 'client_id', 'sentence', 'gender'], axis=1)
y_train = train_merged_df.gender
print(X_train)
X_test = test_merged_df.drop(columns = ['path', 'client_id', 'sentence', 'gender'], axis=1)
y_test = test_merged_df.gender
y_test_array = y_test.to_numpy()

clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


print((y_pred == y_test).sum()/len(y_test))

