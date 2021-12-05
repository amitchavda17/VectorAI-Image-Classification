import kafka
from kafka import KafkaConsumer, KafkaProducer, TopicPartition
from kafka.admin import KafkaAdminClient, NewTopic


class VMessenger:
    def __init__(self, topic="test", server="localhost:9020"):
        """Create a Vmessenger object for sending and receiving messages
        using Apache Kafka message broker

        Args:
            topic (str, optional): [name of topic where messages are sent]. Defaults to "test".
            server (str, optional): [ip address of apache kafka server]. Defaults to "9020".
        """
        self.topic = topic
        self.server = server

    def create_sender(self):
        """Initialize Kafka producer for sending messages"""
        self.producer = KafkaProducer(bootstrap_servers=[self.server])

    def send_message(self, message, topic=None):
        """sends message to selected Kafka topic

        Args:
            message (str): data/message
            topic ([type], optional): [description]. Defaults to None.
        """
        # self.create_sender()
        message = message.encode()
        if not topic:
            topic = self.topic
        self.producer.send(topic, message)
        self.producer.flush()
        # self.producer.close()

    def send_message_batch(self, message_list, topic=None):
        """sends batch of message to kafka topic

        Args:
            message_list (List[str]): list of string messages
            topic (str, optional): topic where message will be sent . Defaults to None.
        """
        self.create_sender()
        if not topic:
            topic = self.topic
        for message in message_list:
            message = message.encode()
            self.producer.send(topic, message)
        self.producer.flush()
        self.producer.close()

    def create_receiver(self, topic=None):
        """Initialze Kaka consumer for reading message stream from topic

        Args:
            topic (str, optional): kafka topic where message is sent. Defaults to None.
        """
        self.consumer = KafkaConsumer(
            bootstrap_servers=[self.server],
            auto_offset_reset="latest",
            group_id="Group2",
            consumer_timeout_ms=1000,
            enable_auto_commit=False,
            auto_commit_interval_ms=1000,
        )
        if not topic:
            topic = self.topic
        topic_partition = TopicPartition(topic, 0)
        assigned_topic = [topic_partition]
        self.consumer.assign(assigned_topic)

    def show_messages(self):
        """Print all messages currently available in topic"""
        for msg in self.consumer:
            print(msg.value.decode())
        self.consumer.commit()

    def read_messages(self):
        """fetch all new messages from topic

        Returns:
            list: list of messages
        """
        messages = []

        for msg in self.consumer:
            messages.append[msg.value.decode()]
        self.consumer.commit()
        return messages

    def shutdown_receiver(self):
        """close Kafka receiver connection"""
        self.consumer.close()

    def create_topic(self, topic):
        self.topic = topic
        admin_client = KafkaAdminClient(
            bootstrap_servers="localhost:9092", client_id="test"
        )
        topic_list = []
        topic_list.append(
            NewTopic(name=self.topic, num_partitions=1, replication_factor=1)
        )
        admin_client.create_topics(new_topics=topic_list, validate_only=False)
