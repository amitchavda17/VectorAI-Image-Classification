import base64
import json
import random

import cv2
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

from VectorAPI.VectorMB import VMessenger


def array_to_base64(arr):
    # arr = arr.tobytes()
    arr_base64 = base64.b64encode(arr)
    return arr_base64


def send_image(
    img_array=None,
    img_path=None,
    topic="ReqQueue",
    server="localhost:9092",
):

    # setup message generator
    sender_clinet = VMessenger(topic, server)
    sender_clinet.create_sender()
    if img_array.all() != None:
        img_array = array_to_base64(img_array)
        print("sending image to kafka...")
        sender_clinet.send_message(img_array, encode=False)
        print("image sent...")


if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # x_train = np.expand_dims(x_train, -1)
    # x_test = np.expand_dims(x_test, -1)
    i = random.randint(0, x_test.shape[0])
    # select a random img
    print(y_test[i])
    img_array = x_test[i].astype("float32")
    print(img_array.shape)
    send_image(img_array=img_array)
