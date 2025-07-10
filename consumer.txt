# consumer.py
from kafka import KafkaConsumer
import json
import requests

consumer = KafkaConsumer(
    'data_stream',  # same as producer
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    group_id='prediction_group',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

# URL of backend API
api_url = "http://localhost:5000/predict"

for message in consumer:
    payload = message.value
    print("Received payload:", payload)
    # Send the payload to backend API using POST request
    try:
        response = requests.post(api_url, json=payload)
        print("API response:", response.json())
    except Exception as e:
        print("Error calling API:", e)
