import kafka
from kafka import KafkaConsumer,KafkaProducer
from kafka.admin import KafkaAdminClient,NewTopic

class VMessenger:
    def __init__(self,topic='test',server=''):
        self.topic=topic
        self.server = server
    
    def create_sender(self):
        self.producer = KafkaProducer(bootstrap_servers=[self.server])
        
    def send_message(self,message,topic=None):
        self.create_sender()
        message = message.encode()
        if not topic:
            topic = self.topic
        self.producer.send(topic,message)
        self.producer.flush()
        self.producer.close()
    
    def send_message_batch(self,message_list,topic=None):
        self.create_sender()
        if not topic:
            topic=self.topic
        for message in message_list:
            message = message.encode()
            self.producer.send(topic,message)
        self.producer.flush()
        self.producer.close()
    def create_receiver(self,topic=None):
        self.consumer = KafkaConsumer(auto_offset_reset='latest',group_id=None,bootstrap_servers=self.server)
        if not topic:
            topic = self.topic
        self.consumer.subscribe([topic])
    
    def read_messages(self):
        for msg in self.consumer:
            print(msg)
    
    def get_messages(self):
        messages = []
        
        for msg in self.consumer:
            messages.append[msg.value.decode()]
        
        return messages
        