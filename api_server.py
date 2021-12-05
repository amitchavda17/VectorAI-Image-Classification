import base64
import json
import random
import asyncio
import cv2
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from io import BytesIO
from VectorAPI.VectorMB import VMessenger
from Vector_ML.Vector_CNN import VectorCNN


def base64_toarr(arr_str):
    arr = base64.decodebytes(arr_str)
    np_arr = np.frombuffer(arr, dtype=np.float32)
    np_arr = np_arr.reshape(28, 28)
    np_arr = np.expand_dims(np_arr, axis=-1)
    return np_arr


async def run_inference(input_img):
    classifier_model = VectorCNN(mode="inference")
    classifier_model.load_model("./snapshots/fmnist_weights.h5")
    y_pred = classifier_model.get_predections(img_array=input_img)
    print("logging model predection", y_pred)


async def get_datastream(topic="ReqQueue", server="localhost:9092"):

    receiver_client = VMessenger(topic=topic, server=server)
    receiver_client.create_receiver()
    while True:
        messages = receiver_client.read_messages()
        if len(messages) > 0:
            for msg in messages:
                arr = base64_toarr(msg)
                print(arr.shape)
                await run_inference(arr)


async def main():

    await get_datastream()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
