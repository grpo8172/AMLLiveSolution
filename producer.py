# producer.py
from kafka import KafkaProducer
import json
import time
import pandas as pd

# Load the dataset
df_full = pd.read_csv("SAML-D.csv", dtype={"Sender_account": "str", "Receiver_account": "str"}, low_memory=True)

# Define the columns to include in the payload (excluding Is_laundering and Laundering_type)
payload_columns = [
    "Time", "Date", "Sender_account", "Receiver_account", "Amount",
    "Payment_currency", "Received_currency", "Sender_bank_location",
    "Receiver_bank_location", "Payment_type"
]

# Sample one row randomly from the dataset
sample_row = df_full.sample(n=1, random_state=42)[payload_columns].iloc[0].to_dict()

# Convert the sample row to a JSON-formatted string
payload_json = json.dumps(sample_row, indent=2)

# Print the JSON payload and example curl command
print("Sample Payload JSON:")
print(payload_json)
print("\nExample curl command:")
print(f"curl -X POST http://localhost:5000/predict -H \"Content-Type: application/json\" -d '{payload_json}'")

# Create a Kafka producer instance
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

topic = 'data_stream'

while True:
    # Send the sample payload as the message
    producer.send(topic, sample_row)
    producer.flush()  # Ensure the data is sent
    print(f"Sent: {sample_row}")
    time.sleep(1)  # Send data every second
