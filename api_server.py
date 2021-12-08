import asyncio
import base64
import json
import random
from io import BytesIO

import cv2
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

from utils.image_conversion import *
from Vector_ML.Vector_CNN import VectorCNN
from VectorAPI.VectorMB import VMessenger

# define class names
labels = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# load classifier model in memory
classifier_model = VectorCNN(mode="inference")
classifier_model.load_model("./snapshots/fmnist_weights.h5")

# create response clinet for sending predections to "Results" topic
response_client = VMessenger(topic="Results")
response_client.create_sender()
# send predections to Results
async def post_results(client=response_client, message=""):
    """Sends model predection to message broker using Vmessage

    Args:
        client (VMessage, optional): response clinet for sending results to "Results" topic.
        Defaults to response_client.
        message (str, optional): model predection. Defaults to "".
    """
    client.send_message(message)
    print("Predections logged to Kafka Result")


async def run_inference(input_img, model=classifier_model):
    """Generates predections on received images using pretrained model

    Args:
        input_img (np.array): imput image
        model (VectorCNN), optional): VectorCNN object with loaded model. Defaults to classifier_model.
    """
    y_pred = model.get_predections(img_array=input_img)
    predection = labels[int(y_pred)]
    print("model predections for the image", predection)
    await post_results(response_client, message=predection)


async def process_datastream(topic="ReqQueue"):
    """Processor data from Request Queue

    Args:
        topic (str, optional): Kafka topic with images. Defaults to "ReqQueue".
    """
    receiver_client = VMessenger(topic=topic)
    receiver_client.create_receiver()
    print("Inference api activated looking for data in Request Queue...")
    while True:
        messages = receiver_client.read_messages()
        if len(messages) > 0:
            for msg in messages:
                print("new image received in Request Queue")
                arr = base64_toarr(msg)
                await run_inference(input_img=arr)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(process_datastream())
