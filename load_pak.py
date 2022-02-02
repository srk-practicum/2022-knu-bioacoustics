import random
import librosa
import os
import csv
import soundfile as sf
from pydub import AudioSegment
f = sf.SoundFile('common_voice_uk_25803029.wav')
print('samples = {}'.format(len(f)))
print('sample rate = {}'.format(f.samplerate))
print('seconds = {}'.format(len(f) / f.samplerate))

i=[]
dict_gender={1:[],0:[]}
tsv_file = open("uk/validated.tsv",encoding='utf-8')
read_tsv = csv.reader(tsv_file, delimiter="\t")
for row in read_tsv:
  #print(row)
    if len(row)>1:
        #print(row[1])
        if(row[6]=="female"):
            dict_gender[1].append(row[1][16:-4])
            i.append(row[1][16:-4])
        if(row[6]=="male"):
            dict_gender[0].append(row[1][16:-4])
            i.append(row[1][16:-4])
for j in range(1842,len(i)):
    try:
        sound = AudioSegment.from_mp3("uk/clips/common_voice_uk_"+i[j]+".mp3")
        sound.export("uk/in_wav/common_voice_uk_"+i[j]+".wav",format="wav")
    except FileNotFoundError:
        print("not good", i[j])
        
