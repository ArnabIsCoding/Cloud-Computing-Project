import os
import csv
import json
import time
import logging
from kafka import KafkaProducer

# Configure structured logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

class FlightDataProducer:
    def __init__(self, bootstrap_servers: list, topic: str):
        self.topic = topic
        logger.info(f"Connecting to Kafka brokers: {bootstrap_servers}")
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    def send_csv_to_kafka(self, file_path: str) -> None:
        """Reads a CSV file and streams each row to Kafka."""
        logger.info(f"Reading {file_path} and streaming to topic: {self.topic}")

        try:
            with open(file_path, mode='r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                count = 0

                for row in csv_reader:
                    self.producer.send(self.topic, value=row)
                    count += 1

                    # Log progress and throttle slightly for local simulation
                    if count % 5000 == 0:
                        logger.info(f"Sent {count} messages...")
                        time.sleep(0.5)

            self.producer.flush()
            logger.info(f"Successfully finished streaming {count} messages to {self.topic}.")

        except FileNotFoundError:
            logger.error(f"Data file not found at {file_path}. Please check your paths.")
        except Exception as e:
            logger.error(f"An error occurred while producing messages: {e}")

if __name__ == "__main__":
    # Environment variables match our docker-compose.yml setup
    KAFKA_BROKER = os.getenv("KAFKA_HOSTS", "localhost:9092").split(",")
    KAFKA_TOPIC = "flight_telemetry"

    # Path to the downloaded Kaggle data
    SAMPLE_FILE = "data/2008.csv"

    producer = FlightDataProducer(bootstrap_servers=KAFKA_BROKER, topic=KAFKA_TOPIC)
    producer.send_csv_to_kafka(SAMPLE_FILE)
