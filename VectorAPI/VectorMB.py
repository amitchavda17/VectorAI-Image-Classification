import kafka
from kafka import KafkaConsumer, KafkaProducer, TopicPartition
from kafka.admin import KafkaAdminClient, NewTopic
from google.cloud import pubsub_v1

class VMessenger:
    def __init__(self, topic="test", server="localhost:9020",broker='kafka'):
        """Create a Vmessenger object for sending and receiving messages
        using Apache Kafka message broker

        Args:
            topic (str, optional): [name of topic where messages are sent]. Defaults to "test".
            server (str, optional): [ip address of apache kafka server]. Defaults to "9020".
        """
        self.topic = topic
        self.server = server
        #defining broker
        self.broker = broker
        self.producer=None
        self.broker=None
        self.consumer=None

    def create_sender(self):
        """Initialize Kafka producer for sending messages"""
        if broker=='kafka':
            self.producer = KafkaProducer(bootstrap_servers=[self.server])
        else:
            self.producer= pubsub_v1.PublisherClient()


    def send_message(self, message, topic=None, encode=True):
        """sends message to selected Kafka topic

        Args:
            message (str): data/message
            topic ([type], optional): [description]. Defaults to None.
        """
        #self.create_sender()

        if encode:
            message = message.encode()
        if not topic:
            topic = self.topic
        if self.producer=='kafka':
            self.producer.send(topic, message)
            self.producer.flush()
        #self.producer.close()
        else:
            #use pubsub
            topic_name = 'projects/{project_id}/topics/{topic}'.format(
            project_id=os.getenv('GOOGLE_CLOUD_PROJECT'),
            topic=topic,  # Set this to something appropriate.
            )
            future = self.producer.publish(topic_name,message)
            print('published msg id ',future.results())



    def send_message_batch(self, message_list, topic=None, encode=True):
        """sends batch of message to kafka topic

        Args:
            message_list (List[str]): list of string messages
            topic (str, optional): topic where message will be sent . Defaults to None.
        """
        if not topic:
            topic = self.topic
        
        for message in message_list:
            if encode:
                message = message.encode()
            if self.broker=='kafka':
                self.producer.send(topic, message)
            else:
                #use pubsub 
                future = self.producer.publish(topic,message)
        if self.broker=='kafka':
            self.producer.flush()

    def create_receiver(self, topic=None):
        """Initialze Kaka consumer for reading message stream from topic

        Args:
            topic (str, optional): kafka topic where message is sent. Defaults to None.
        """
        if not topic:
            topic = self.topic

        if self.broker == 'kafka':
            self.consumer = KafkaConsumer(
                bootstrap_servers=[self.server],
                auto_offset_reset="latest",
                group_id="Group2",
                consumer_timeout_ms=1000,
                enable_auto_commit=False,
                auto_commit_interval_ms=1000,
            )
           
            topic_partition = TopicPartition(topic, 0)
            assigned_topic = [topic_partition]
            self.consumer.assign(assigned_topic)
        else:
            self.consumer = pubsub_v1.SubscribeClient()
            self.subscription_path= 'projects/{project_id}/subscriptions/{topic}'.format(
            project_id=os.getenv('GOOGLE_CLOUD_PROJECT'),
            topic=topic,  # Set this to something appropriate.
            )

    

    def show_messages(self):
        """Print all messages currently available in topic"""
        if self.broker =='kafka':
            for msg in self.consumer:
                print(msg.value.decode())
            self.consumer.commit()
        else:
            response = self.consumer.pull(
            request={
                "subscription": self.subscription_path
            })
            for msg in response.received_messages:
                print(msg.message.data.decode())


    def read_messages(self):
        """fetch all new messages from topic

        Returns:
            list: list of messages
        """
        messages = []
        if self.broker=='kafka':
            for msg in self.consumer:
                messages.append(msg.value)
            self.consumer.commit()
        else:
            response = self.consumer.pull(
            request={
                "subscription": self.subscription_path
            })
            for msg in response.received_messages:
                messages.append(msg.message.data.decode())
            ack_ids = [msg.ack_id for msg in response.received_messages]
            subscriber.acknowledge(
                request={
                    "subscription": subscription_path,
                    "ack_ids": ack_ids,
                })

        return messages

    def shutdown_receiver(self):
        """close Kafka receiver connection"""
        if self.producer=="kafka":
            self.consumer.close()

    def create_topic(self, topic):
        if self.broker == "kafka":
            self.topic = topic
            admin_client = KafkaAdminClient(
                bootstrap_servers="localhost:9092", client_id="test"
            )
            topic_list = []
            topic_list.append(
                NewTopic(name=self.topic, num_partitions=1, replication_factor=1)
            )
            admin_client.create_topics(new_topics=topic_list, validate_only=False)
    
