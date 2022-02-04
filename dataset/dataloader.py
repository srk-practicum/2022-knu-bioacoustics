import os
import numpy as np
import librosa as lr
from scipy import signal, fft as scfft
import pandas as pd
from pydub import AudioSegment
from sklearn.preprocessing import LabelEncoder
import pickle
import pywt


class Dataloader:

    def __init__(self, sample_rate, duration, random_seed=10):
        """
        DataLoader object.
        :param sample_rate: sample rate (Hz) of all audios form dataset.
        :param duration: duration (seconds) of all audios from dataset.
        :param random_seed: random seed for numpy random generator.
        """
        np.random.seed(random_seed)
        self.random_seed = random_seed
        self.sample_rate = sample_rate
        self.duration = duration

        # calculate samples for each audio
        self.audio_samples = sample_rate * duration

        # store audios
        self.audios = []

        # store sample rates for resampling
        self.sample_rates = []

        # store classes for classification
        self.target = []

        # encoder to encode categorical target variable into integers.
        self.le = LabelEncoder()

    def fit(self, folderpath, filename, target_categorical):
        """
        Dataset preprocessing.
        :param folderpath: path to dataset folder with tsv files and clips folder.
        :param filename: name of tsv file.
        :param target_categorical: target, that must be predicted by classifiers.
        :return: cache: energies for all audio, for reproducing initial audios' energy.
        """
        self.load(folderpath, filename, target_categorical)
        self.resample_dataset()
        self.time_normalize()
        return self.energy_normalize()

    def shuffle(self):
        """
        Shuffling of all dataset.
        Must be used after all preprocessing, because it shuffles only audios and targets,
        concatenating them and then separating them back.
        :return: None
        """
        np.random.seed(self.random_seed)
        self.target = np.array(self.target)
        dataset = np.concatenate([self.audios, self.target.reshape(-1, 1)], axis=1)
        np.random.shuffle(dataset)
        self.audios = dataset[:, :-1]
        self.target = dataset[:, -1:].reshape(-1)

    @staticmethod
    def unison_shuffled_copies(a, b):
        assert len(a) == len(b), 'arrays must be the same length'
        idx = np.random.permutation(len(a))
        return a[idx], b[idx]

    def train_test_split(self, dataset=None, test_coefficient=0.2):
        """
        Splitting dataset into 2 datasets: train and test.
        :param dataset: dataset (features).
        :param labels: categorical variable.
        :param test_coefficient: test part of dataset, in percents.
        :return: train data, test data - tuples.
        """

        if dataset is None:
            dataset = self.audios
        labels = np.array(self.target)

        samples = dataset.shape[0]
        test_examples = int(test_coefficient * samples)
        train_examples = samples - test_examples

        dataset, labels = self.unison_shuffled_copies(dataset, labels)

        train_audios = dataset[:train_examples]
        test_audios = dataset[train_examples:]
        train_labels = labels[:train_examples]
        test_labels = labels[train_examples:]

        return (train_audios, train_labels), (test_audios, test_labels)

    @staticmethod
    def prepare_shape(dataset):
        if dataset.ndim >= 3:
            dataset = dataset.reshape(dataset.shape[0], -1)
        return dataset

    def load(self, folderpath, filename, target_categorical):
        """
        Dataset loading.
        :param folderpath: path to dataset folder with tsv files and clips folder.
        :param filename: name of tsv file.
        :param target_categorical: target, that must be predicted by classifiers.
        :return: None
        """
        # reading tsv file
        df = pd.read_csv(os.path.join(folderpath, filename), sep="\t")
        # clearing all nan values for current target
        notna_df = df[df[target_categorical].notna()]

        # folder for storing converted to wzv audio files
        wav_folder_path = os.path.join(folderpath, 'converted')

        # creating this folder if it not exists
        try:
            os.stat(wav_folder_path)
        except Exception as e:
            print(e)
            os.mkdir(wav_folder_path)

        for audio_name, label in zip(notna_df['path'], notna_df[target_categorical]):
            # loading mp3 audio
            sound = AudioSegment.from_mp3(os.path.join(folderpath, 'clips', audio_name))
            # path for wav file
            current_wav_path = os.path.join(wav_folder_path, audio_name[:-3] + 'wav')
            # exporting to wav
            sound.export(current_wav_path, format='wav')

            # loading wav file. You may set sr=self.sample rate here and then don't need to
            # resample audios after loading. By default, sr=22050. If you set None,
            # librosa will load audio with original sample rate.
            y, sr = lr.load(current_wav_path, sr=None)

            self.audios.append(y)
            self.target.append(label)
            self.sample_rates.append(sr)

            print(f"audio {audio_name} loaded.")

        # encoding loaded target variable into integers
        self.target = self.le.fit_transform(self.target)

    def time_normalize(self, shift=0):
        """
        Normalizing by time. Cutting audios.
        :param shift: if not 0, it will cut self.audio_samples samples, shifted by shift.
        :return: Тщту
        """
        for i, audio in enumerate(self.audios):
            # cut audio for time normalization
            self.audios[i] = audio[shift:shift + self.audio_samples]
            print(f"audio {i} normalized by time.")

    def energy_normalize(self):
        """
        Normalizing by energy. After this, energy of all signals will be one.
        :return: initial energies.
        """
        energies = []
        for i, audio in enumerate(self.audios):
            # calculate audio energy
            current_energy = np.sum(audio ** 2)
            energies.append(current_energy)
            # normalize by energy - divide by sqrt(energy), then the energy of
            # normalized audio will be one.
            self.audios[i] = audio / np.sqrt(current_energy)
            print(f"audio {i} normalized by energy.")
        return energies

    def resample_dataset(self):
        """
        Resampling of all audios.
        :return: None
        """
        for i, audio in enumerate(self.audios):
            # using stored sample rates, resample each audio to given self.sample_rate
            self.audios[i] = lr.resample(audio, self.sample_rates[i], self.sample_rate)
            print(f"audio {i} resampled.")

    def save(self, filepath):
        """
        Saving audios to pkl file.
        :param filepath: path to pkl file
        :return: None
        """
        data = {
            'audios': self.audios,
            'target': self.target,
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'encoder': self.le
        }
        with open(os.path.join(filepath), 'wb') as f:
            try:
                pickle.dump(data, f)
            except Exception as e:
                print(e)
                return

    def load_pkl(self, filepath):
        """
        Loading dataset from pkl file.
        :param filepath: path to pkl file
        :return: None
        """
        try:
            data = pickle.load(open(filepath, 'rb'))
            self.sample_rate = data['sample_rate']
            self.duration = data['duration']
            self.audios = data['audios']
            self.target = data['target']
            self.le = data['encoder']
        except Exception as e:
            print(e)
            data = None

    def stft(self, window, nperseg=200, noverlap=100):
        """
        Short-Time Fourier Transform.
        :param window: str, for example 'hamming', 'bartlett' etc.
        :param nperseg: length of window.
        :param noverlap: length of overlapping windows' section.
        :return: for each audio: frequencies scale, times scale, transformed audios.
        """
        frequencies = []
        times = []
        stft = []
        for audio in self.audios:
            freq, t, transformed = signal.stft(audio,
                                               self.sample_rate,
                                               window=window,
                                               nperseg=nperseg,
                                               noverlap=noverlap)
            frequencies.append(freq)
            times.append(t)
            stft.append(np.abs(transformed))
        return np.array(frequencies), np.array(times), np.array(stft)

    def features(self,
                 n_fft=512,
                 hop_length=384,
                 n_mfcc=256,
                 features=None
                 ):
        """
        Feature extraction: mel spectrogram, mel frequency cepstral coefficients.
        :param n_fft: length of the FFT window.
        :param hop_length: number of audio samples between adjacent STFT columns.
        :param n_mfcc: number of MFCCs to return.
        :param features: features to extract, iterable. Example: ['mfcc'].
        Possible: ['mfcc', 'melspec', 'wavelets'].
        :return:
        """

        output = dict()
        if 'mfcc' in features:
            mfcc = []
            for audio in self.audios:
                mfcc.append(lr.feature.mfcc(y=np.nan_to_num(audio),
                                            sr=self.sample_rate,
                                            n_mfcc=n_mfcc,
                                            n_fft=n_fft,
                                            hop_length=hop_length))
            output['mfcc'] = np.array(mfcc)

        if 'melspec' in features:
            melspec = []
            for audio in self.audios:
                melspec.append(lr.feature.melspectrogram(y=np.nan_to_num(audio),
                                                         sr=self.sample_rate,
                                                         n_fft=n_fft,
                                                         hop_length=hop_length))
            output['melspec'] = np.array(melspec)

        if 'wavelets' in features:
            wavelets = []
            for audio in self.audios:
                ca, cd = pywt.dwt(audio, 'db2')
                f = np.array([ca, cd]).reshape(-1)
                wavelets.append(f)
            output['wavelets'] = np.array(wavelets)

        return output

    def create_images(self, output):
        pass