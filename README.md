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
   - We use ```kafka-python``` library for communicating with Apache Kafka server
   - Creating ```VectorAPI``` for simplying message transfers via Kafka
   ```python
   from VectorAPI.VectorMB import VMessenger

   client = VMessenger(topic,server_url)
   ```  
   for more details refer API source code and ```Message_broker_api.ipynb``` for demo [clickhere](Messege_broker_api.ipynb)

    
C) Serve ML models using message broking
    #TODO
    