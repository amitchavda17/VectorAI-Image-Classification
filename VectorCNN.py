import os
import matplotlib.pyplot as plt
import cv2
import random
import collections
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
class VectorCNN:
    def __init__(self,x_train,y_train,x_test,y_test,labels,no_classes,mode='train'):
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
            self.optimizer = None
            self.img_height = x_train.shape[1]
            self.img_width = x_train.shape[2]
            self.channels = x_train.shape[3]
        elif self.model=='inference':
            self.weights =
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
           




    