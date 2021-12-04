import os
import matplotlib.pyplot as plt
import random
import collections
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class VectorCNN:
    def __init__(self,x_train,y_train,x_test,y_test,labels,no_classes,weightsmode='train'):
        self.mode = mode
        if self.model=='train'
            self.x_train = x_train
            self.y_train = y_train
            self.x_test = x_test
            self.y_test = y_test
            self.lables = labels
            self.no_classes = classes
            assert len(x_train.shape)==4
            assert len(x_test.shape)==4
            self.model = None
            self.callbacks =None
            self.img_height = x_train.shape[1]
            self.img_width = x_train.shape[2]
            self.channels = x_train.shape[3]
        elif self.model=='inference':
            self.weights = ''
            self.model =
            self.x_test =x_test
            self.y_test = y_test

    def prepare_data(self,verbose=0):
        
        self.x_train =self. x_train.astype("float32") / 255.
        self.x_train = np.expand_dims(self.x_train, -1)
        self.y_train = to_categorical(self.y_train,10)
        self.x_test = self.x_test.astype("float32") / 255.
        self.x_test = np.expand_dims(self.x_test, -1)
        #coverting classes to binary coding
        self.y_test =to_categorical(self.y_test,10)
        if verbose>0:
            print("x_train shape:", self.x_train.shape)
            print(self.x_test.shape[0], "test samples")
            print(self.x_train.shape[0], "train samples")

    def prepare_test_data(self):
        self.x_test = self.x_test.astype("float32") / 255.
        self.y_test =to_categorical(self.y_test,10)

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', 
                        data_format='channels_last', input_shape=(self.img_height,self.img_width,self.channels)))
        model.add(BatchNormalization())

        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', 
                        data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=1, padding='same', 
                        data_format='channels_last'))
        model.add(MaxPooling2D(pool_size=(2, 2)))  
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', strides=1, padding='same', 
                        data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        self.model = model

    def compile_model(self,loss_fn='categorical_crossentropy',metric='accuracy',optimizer_fn='',learning_rate =0.001,callbacks=True):
        if optimizer_fn=='':
            adam = optimizers.Adam(learning_rate = learning_rate, beta_1 = 0.9, beta_2 = 0.99, epsilon = 1e-8)
        if type(metric) is str:
            metric = [metric]

        self.model.compile(loss=loss_fn,optimizer=adam,metrics=[metric])

        if callabacks:
            target_dir = './snapshots'

            if not os.path.exists(target_dir):
                os.mkdir(target_dir)

        file_path = 'snapshots/best_weight{epoch:03d}.h5'
        checkpoints = callbacks.ModelCheckpoint(file_path, monitor = 'val_loss', verbose = 0, save_best_only = True, mode = 'auto')
        lr_op = callbacks.ReduceLROnPlateau(monitor = "val_loss", factor = 0.1, patience = 5,
                    verbose = 1, mode = "auto", cooldown = 0,min_lr = 1e-30)
        early_stop = callbacks.EarlyStopping(
            monitor = "val_loss",
            min_delta = 0.00001,
            patience = 9,
            verbose = 1,
            mode = "auto",
            baseline = None,
            restore_best_weights = False,
        )
        self.callbacks = [ checkpoints, lr_op, early_stop]

    def train_model(self,batch_size=32,epochs=50):

        datagen = _get_generator():
        self.history = self.model.fit(datagen.flow(x_train, y_train, batch_size = batch_size), 
            epochs=epochs, verbose=1, callbacks=self.callbacks, validation_data=(x_test, y_test),validation_batch_size=batch_size)
    
    def _get_generator(self):
        datagen = ImageDataGenerator(
        rotation_range = 8,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        shear_range = 0.3,# shear angle in counter-clockwise direction in degrees  
        width_shift_range=0.08,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.08,  # randomly shift images vertically (fraction of total height)
        vertical_flip=True)  # randomly flip images

        datagen.fit(self.x_train)
        
        return datagen
    
    def load_model(self,weights_path):
        self.model.load_weights(weights_path)
   
    def evaluate_model(self):
        loss,acc=self.model.evaluate(x_test,yest)
        print('Test loss',loss)
        print('Test acc',acc)

    def get_predections(self,img_array=None,img_path=None,img_height,img_width):
        if img_array!=None:
            x = x.astype('float32')/255.
            y_pred = self.model.predict(x)
            y_pred = np.argmax(y_pred,axis=1)
            
            return y_pred

        elif img_path!=None:
            img = load_img(img_path, target_size=(img_height,img_width))
            img_tensor = img_to_array(img)
            img_tensor = np.expand_dims(img_tensor, axis=0)   
            img_tensor/=255.0
            y_pred = model.predict(img_tensor)
            y_pred = np.argmax(y_pred,axis=1)
            
            return y_pred



        





    