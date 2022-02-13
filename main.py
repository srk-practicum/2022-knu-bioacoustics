
import kivy
import random

from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
import time
import soundfile
import sounddevice as sd

import librosa
import pandas as pd
from pydub import AudioSegment
import scipy as sp
import pywt
from sklearn.ensemble import RandomForestClassifier
import csv
import numpy as np

from kivy.uix.image import Image
from kivy.graphics import Color, Rectangle

blue = [0, 0, 1, 1]

#забираем фичи

def get_features(y):
    stft = np.abs(librosa.stft(y))

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=16000).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y,sr=16000).T,axis=0)
    cA, cD = pywt.dwt(y, 'db2')
    return [*chroma, *mel, cA.mean(), cD.mean()]


def get_controls_feature(file):
    res_cont = []

    res_cont.append(get_features(librosa.load(file, sr = 16000)[0]))
    print( get_features(librosa.load(file, sr = 16000)[0]   ))

    return res_cont



class MainApp(App):
    def build(self):
        img = Image(source='meh.jpg',
                    size_hint=(1, .5),
                    pos_hint={'center_x': .5, 'center_y': .5})
        button = Button(text='Record',
                        size_hint=(.5, .5),
                        pos_hint={'center_x': .5, 'center_y': .5})
        button.bind(on_press=self.on_press_button)
        self.text=button.text
        #button.bind(on_press=self.on_solution)
        return button

    def on_press_button(self, instance):
        current = self.text
        print('Вы нажали на кнопку!', current)
        if current=="Record":
            sd.default.samplerate = 16000
            sd.default.channels = 1
            duration = 1
            fs = 16000
            time.sleep(1)
            my_sound = sd.rec(int(duration * fs))
            sd.sleep(int(3000))
            # sd.wait()
            soundfile.write('hellp.wav', my_sound, 16000)
            self.text="Analyse"
            instance.text = "Analyse"
        elif current == "Analyse":
            merged_df = pd.read_csv('data_hand_after.csv')
            y = merged_df.target.values

            X = merged_df.drop(columns=['path', 'target', 'Unnamed: 0'], axis=1).values

            X_for_test = get_controls_feature('hellp.wav')
            clf = RandomForestClassifier()
            clf.fit(X, y)
            y_pred = clf.predict(X_for_test)
            print(y_pred)
            self.text = "Own sound"*y_pred[0]+"Other sound"*(not y_pred[0])
            instance.text = "Own sound"*y_pred[0]+"Other sound"*(not y_pred[0])





if __name__ == "__main__":
    app = MainApp()
    #app2=HBoxLayoutExample()
    #app2.run()
    app.run()
