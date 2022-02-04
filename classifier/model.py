import numpy as np
import tensorflow as tf

from dataset.dataloader import Dataloader
from sklearn.metrics import accuracy_score, confusion_matrix


class Model:

    def __init__(self, dataset: Dataloader):
        self.dataset = dataset

    def reshape(self, features):
        return [feature.reshape(-1) for feature in features]

    def train(self, model, test_coef, features='audios'):

        assert features in ['audios', 'mfcc', 'melspec', 'stft', 'wavelets'], \
            "features must be a string from ['audios', 'mfcc', 'melspec', 'stft', 'wavelets']"

        (x_train, y_train), (x_test, y_test) = (None, None), (None, None)

        if features == 'audios':
            (x_train, y_train), (x_test, y_test) = self.dataset.train_test_split(test_coefficient=test_coef)
        elif features == 'mfcc':
            mfcc = self.dataset.features(features=['mfcc'])['mfcc']
            mfcc = self.dataset.prepare_shape(mfcc)
            (x_train, y_train), (x_test, y_test) = self.dataset.train_test_split(dataset=mfcc,
                                                                                 test_coefficient=test_coef)
        elif features == 'melspec':
            melspec = self.dataset.features(features=['melspec'])['melspec']
            melspec = self.dataset.prepare_shape(melspec)
            (x_train, y_train), (x_test, y_test) = self.dataset.train_test_split(dataset=melspec,
                                                                                 test_coefficient=test_coef)
        elif features == 'stft':
            _, __, stft = self.dataset.stft('cosine')
            np.nan_to_num(stft, copy=False)
            (x_train, y_train), (x_test, y_test) = self.dataset.train_test_split(dataset=stft,
                                                                                 test_coefficient=test_coef)

        elif features == 'wavelets':
            wavelets = self.dataset.features(features=['wavelets'])['wavelets']
            np.nan_to_num(wavelets, copy=False)
            (x_train, y_train), (x_test, y_test) = self.dataset.train_test_split(dataset=wavelets,
                                                                                 test_coefficient=test_coef)

        x_train = self.reshape(x_train)
        x_test = self.reshape(x_test)

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        output = {
            'model': model,
            'pred': predictions,
            'true': y_test
        }

        return output

    def predict(self, model, x):
        return model.predict(x)

    def estimate(self, true, pred):
        print(f"Accuracy: {accuracy_score(true, pred)}")
        print(f"Confusion matrix: {confusion_matrix(true, pred)}")

    @staticmethod
    def show_nn_layers(model):
        print('---------------MODEL---------------')
        for i, layer in enumerate(model.layers):
            print(i, layer.name)
        print('-----------------------------------')

    def prepare_net(self,
                    base_model,
                    data_shape,
                    learning_rate=0.0001,
                    layers_to_freeze=None,
                    show_model=False):

        # create a pre-trained model, adding top layer with shape of given data
        base_model = base_model(input_tensor=tf.keras.layers.Input(shape=data_shape),
                                weights='imagenet',
                                include_top=False)
        # add a global spatial average pooling layer
        x = base_model.output
        x_pool = tf.keras.layers.GlobalAveragePooling2D()(x)
        # add logistic layer (we have only 2 classes for our binary classification)
        final_pred = tf.keras.layers.Dense(2, activation='softmax')(x_pool)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=final_pred)

        if layers_to_freeze is None:
            for layer in model.layers:
                layer.trainable = False
        else:
            for layer in model.layers[:layers_to_freeze]:
                layer.trainable = False
            for layer in model.layers[layers_to_freeze:]:
                layer.trainable = True

        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        if show_model:
            self.show_nn_layers(model)

        return model

    def fit_net(self,
                base_model,
                features,
                test_coef=0.2,
                batch_size=64,
                epochs=10,
                learning_rate=0.0001,
                layers_to_freeze=None,
                show_model=False
                ):

        assert features in ['audios', 'mfcc', 'melspec', 'stft'], \
            "features must be a string from ['audios', 'mfcc', 'melspec', 'stft']"

        (x_train, y_train), (x_valid, y_valid) = (None, None), (None, None)

        if features == 'audios':
            (x_train, y_train), (x_valid, y_valid) = self.dataset.train_test_split(test_coefficient=test_coef)
        elif features == 'mfcc':
            mfcc = self.dataset.features(features=['mfcc'])['mfcc']
            (x_train, y_train), (x_valid, y_valid) = self.dataset.train_test_split(dataset=mfcc,
                                                                                   test_coefficient=test_coef)
        elif features == 'melspec':
            melspec = self.dataset.features(features=['melspec'])['melspec']
            (x_train, y_train), (x_valid, y_valid) = self.dataset.train_test_split(dataset=melspec,
                                                                                   test_coefficient=test_coef)
        elif features == 'stft':
            _, __, stft = self.dataset.stft('cosine')
            np.nan_to_num(stft, copy=False)
            (x_train, y_train), (x_valid, y_valid) = self.dataset.train_test_split(dataset=stft,
                                                                                   test_coefficient=test_coef)

        data_shape_unit = tuple(np.append(x_train.shape[1:], 1))
        data_shape = tuple(np.append(x_train.shape[1:], 3))

        x_train = x_train.reshape((-1,) + data_shape_unit)
        x_train = np.concatenate([x_train for _ in range(3)], axis=3)
        x_valid = x_valid.reshape((-1,) + data_shape_unit)
        x_valid = np.concatenate([x_valid for _ in range(3)], axis=3)
        y_train = y_train.reshape(-1, 1)
        y_valid = y_valid.reshape(-1, 1)

        model = self.prepare_net(base_model,
                                 data_shape,
                                 learning_rate=learning_rate,
                                 layers_to_freeze=layers_to_freeze,
                                 show_model=show_model)

        return model.fit(x=x_train,
                         y=y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         validation_data=(x_valid, y_valid))
