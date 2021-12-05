import base64
import json
import random

import cv2
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

from VectorAPI.VectorMB import VMessenger
from utils.image_conversion import *


def send_image(
    img_array=None,
    img_path=None,
    topic="ReqQueue",
    server="localhost:9092",
):
    """Send images to Kafka for inference

    Args:
        img_array (np.array, optional): image array. Defaults to None.
        img_path ([type], optional): path to img_file. Defaults to None.
        topic (str, optional): Kafka topic. Defaults to "ReqQueue".
        server (str, optional): Kafka server url. Defaults to "localhost:9092".
    """

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
