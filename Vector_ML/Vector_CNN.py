import collections
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    array_to_img,
    img_to_array,
)
from tensorflow.keras.utils import to_categorical


class VectorCNN:
    def __init__(
        self,
        x_train=None,
        y_train=None,
        x_test=None,
        y_test=None,
        labels=[],
        no_classes=None,
        weights=None,
        mode="train",
    ):
        self.mode = mode
        if self.mode == "train":
            self.x_train = x_train
            self.y_train = y_train
            self.x_test = x_test
            self.y_test = y_test
            self.lables = labels
            self.no_classes = no_classes
            assert len(x_train.shape) == 4
            assert len(x_test.shape) == 4
            self.model = None
            self.train_callbacks = None
            self.img_height = x_train.shape[1]
            self.img_width = x_train.shape[2]
            self.channels = x_train.shape[3]
        elif self.mode == "inference":
            self.weights = weights
            self.model = None
            self.x_test = x_test
            self.y_test = y_test

    def prepare_data(self, verbose=0):

        self.x_train = self.x_train.astype("float32") / 255.0
        self.y_train = to_categorical(self.y_train, 10)
        self.x_test = self.x_test.astype("float32") / 255.0
        self.y_test = to_categorical(self.y_test, 10)
        if verbose > 0:
            print("x_train shape:", self.x_train.shape)
            print(self.x_test.shape[0], "test samples")
            print(self.x_train.shape[0], "train samples")

    def prepare_test_data(self):
        self.x_test = self.x_test.astype("float32") / 255.0
        self.y_test = to_categorical(self.y_test, 10)

    def create_model(self):
        model = Sequential()
        model.add(
            Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation="relu",
                strides=1,
                padding="same",
                data_format="channels_last",
                input_shape=(self.img_height, self.img_width, self.channels),
            )
        )
        model.add(BatchNormalization())

        model.add(
            Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation="relu",
                strides=1,
                padding="same",
                data_format="channels_last",
            )
        )
        model.add(BatchNormalization())
        model.add(
            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation="relu",
                strides=1,
                padding="same",
                data_format="channels_last",
            )
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(
            Conv2D(
                filters=128,
                kernel_size=(3, 3),
                activation="relu",
                strides=1,
                padding="same",
                data_format="channels_last",
            )
        )
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(128, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(10, activation="softmax"))

        self.model = model

    def compile_model(
        self,
        loss_fn="categorical_crossentropy",
        metric="accuracy",
        optimizer_fn="",
        learning_rate=0.001,
        callback_flag=True,
    ):
        if optimizer_fn == "":
            adam = optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=0.9,
                beta_2=0.99,
                epsilon=1e-8,
            )
        if type(metric) is str:
            metric = [metric]

        self.model.compile(loss=loss_fn, optimizer=adam, metrics=[metric])

        if callback_flag:
            target_dir = "./snapshots"
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)

            file_path = "snapshots/best_weight{epoch:03d}.h5"
            checkpoints = callbacks.ModelCheckpoint(
                file_path,
                monitor="val_loss",
                verbose=0,
                save_best_only=True,
                mode="auto",
            )
            lr_op = callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=5,
                verbose=1,
                mode="auto",
                cooldown=0,
                min_lr=1e-30,
            )
            early_stop = callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0.00001,
                patience=9,
                verbose=1,
                mode="auto",
                baseline=None,
                restore_best_weights=False,
            )
        self.train_callbacks = [checkpoints, lr_op, early_stop]

    def train_model(self, batch_size=32, epochs=50):

        datagen = self._get_generator()
        self.history = self.model.fit(
            datagen.flow(self.x_train, self.y_train, batch_size=batch_size),
            epochs=epochs,
            verbose=1,
            callbacks=self.train_callbacks,
            validation_data=(self.x_test, self.y_test),
            validation_batch_size=batch_size,
        )

    def _get_generator(self):
        datagen = ImageDataGenerator(
            rotation_range=8,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.1,  # Randomly zoom image
            shear_range=0.3,  # shear angle in counter-clockwise direction in degrees
            width_shift_range=0.08,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.08,  # randomly shift images vertically (fraction of total height)
            vertical_flip=True,
        )  # randomly flip images

        datagen.fit(self.x_train)

        return datagen

    def load_model(self, weights_path):
        self.model = load_model(weights_path)

    def evaluate_model(self, x=None, y=None):
        if x == None and y == None:
            self.prepare_test_data()
            loss, acc = self.model.evaluate(self.x_test, self.y_test)
            print("Test loss", loss)
            print("Test acc", acc)
        else:
            x = self.x.astype("float32") / 255.0
            y = to_categorical(self.y, 10)
            loss, acc = self.model.evaluate(x, y)
            print("Test loss", loss)
            print("Test acc", acc)

    def get_predections(
        self, img_array=None, img_path=None, img_height=28, img_width=28
    ):
        if img_array.all() != None:
            x = img_array.astype("float32") / 255.0
            x = np.expand_dims(x, axis=0)
            y_pred = self.model.predict(x)
            y_pred = np.argmax(y_pred, axis=1)

            return y_pred

        elif img_path != None:
            img = load_img(img_path, target_size=(img_height, img_width))
            img_tensor = img_to_array(img)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            img_tensor /= 255.0
            y_pred = self.model.predict(img_tensor)
            y_pred = np.argmax(y_pred, axis=1)

            return y_pred
        else:
            x = self.x_test.astype("float32") / 255.0
            y_pred = self.model.predict(x)
            y_pred = np.argmax(y_pred, axis=1)
            return y_pred
