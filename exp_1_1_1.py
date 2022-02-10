import random
import librosa
import os
import csv
import soundfile as sf
from pydub import AudioSegment
import random
from sklearn import svm
f = sf.SoundFile('common_voice_uk_25803029.wav')
print('samples = {}'.format(len(f)))
print('sample rate = {}'.format(f.samplerate))
print('seconds = {}'.format(len(f) / f.samplerate))

i=[]
i_g=[]
tsv_file = open("uk/validated.tsv",encoding='utf-8')
read_tsv = csv.reader(tsv_file, delimiter="\t")
for row in read_tsv:
  #print(row)
    if len(row)>1:
        #print(row[1])
        if(row[6]=="female"):
            i_g.append(1)
            i.append(row[1][16:-4])
        if(row[6]=="male"):
            i_g.append(0)
            i.append(row[1][16:-4])

x, sr = librosa.load('common_voice_uk_25803029.wav',
                     sr = 16000,
                     mono=True,
                     offset=0.0,
                     duration=2.0)
len_i=14000
N_v=1000
d=set()
for ii in range(N_v):
    rand_num=random.randint(0, len_i)
    while(rand_num in d):
        rand_num=random.randint(0, len_i)
    d.add(rand_num)

for ii in d:
    x, sr = librosa.load("uk/in_wav/common_voice_uk_"+i[ii]+".wav",
                     sr = 16000,
                     mono=True,
                     offset=0.0,
                     duration=2.0)
    
    
    
    
