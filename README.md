#Vector image classification API 
make a virtual enviornment and install the requirements
```
pip install -r requirements.txt
```
A) Build Image Classifier and Library
1) Fashion MNIST image classifier :
   Trained a custom image classifier to classify fashion mnist images please refer
   Refer ```Fashion_mnist.ipynb``` [clickhere](Fashion_mnist.ipynb)


2)  Vector_ML  Library/API to automate building and testing Classification Model:
    ```python
    Vector_ML\Vector_CNN
    
    class VectorCNN : library for building image classification which uses tf.keras
    ```
    Launch docsumo_text_classification.ipynb 
3) Testing the Vector_ML Library to build models:
    Refer ```mnist_classifier_vectorMl.ipynb``` [clickhere](mnist_classifier_vectorML.ipynb)

B) API to send receive messages via message brokers:

   - Install Apache Kafka on your localsystem and run on ```localhost:9029``` 
     or Setup pubsub from google
   - We use ```kafka-python``` library for communicating with Apache Kafka server
   - we use gcloud python library for communicating with pubsub
   - Creating ```VectorAPI``` for simplying message transfers via Kafka
    **note : the system by default uses kafka but pubsub can be used by selecting the broker as pubsub while initializing VMessager**
   ```python
   from VectorAPI.VectorMB import VMessenger

   client = VMessenger(topic,server_url)
   ```  
   for more details refer API source code and ```Message_broker_api.ipynb``` for demo [clickhere](Messege_broker_api.ipynb)

    
C) Serve ML models using message broking
   - Client : sends images to kafka for processing
   ```client.py``` randomly picks images from fashion_mnist datasets and sends to Kafka **ReqQueue** for inference

   - Server : Fetches data from Kafka and runs infenrece using pretrained model 
    i)```api_server.py``` runs  process_stream_data which keeps looking for new data in topic and sends the received images for predections in async mode 
        ii) Model predections are printed in terminal as logs
        iii) After that model predections are sent to 'Results' topic on Kafka which can be fetched by any subsriber.
        iv) start inference application to fetch and proces images
        ```
        python api_server.py
        ```
        v) send data for inference to Kafka
        ```
        python client.py
        ```

    


   
    