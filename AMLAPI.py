import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from kafka import KafkaProducer
import json

app = Flask(__name__)
CORS(app)

# Loading the trained models and feature list
model = joblib.load('random_forest_model.pkl')
pca = joblib.load('pca_model.pkl')
trained_features = joblib.load('trained_features.pkl')
encoder = joblib.load('encoder.pkl')  # Load the OrdinalEncoder used in training


@app.route('/predict', methods=['POST'])
def predict():
    categorical_columns = [
        "Time", "Date", "Sender_account", "Receiver_account", "Amount",
        "Payment_currency", "Received_currency", "Sender_bank_location",
        "Receiver_bank_location", "Payment_type"
    ]

    try:
        # Get the input data
        data = request.get_json()
        df = pd.DataFrame([data])  # Convert input data to DataFrame

        # Handle missing values
        df.fillna("Unknown", inplace=True)

        # Apply ordinal encoding to high cardinality columns (like 'Sender_account', 'Receiver_account')
        high_cardinality_cols = ["Sender_account", "Receiver_account"]
        df[high_cardinality_cols] = encoder.transform(df[high_cardinality_cols])

        # Convert categorical columns to category dtype for efficient encoding
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype("category")

        # One-hot encode categorical columns (same as in training phase)
        df_encoded = pd.get_dummies(df, columns=categorical_columns, dtype="int8")

        # Ensure the input data has the same features as during training (columns must match)
        missing_cols = [feature for feature in trained_features if feature not in df_encoded.columns]

        if missing_cols:
            missing_data = pd.DataFrame(0, index=df_encoded.index, columns=missing_cols)
            df_encoded = pd.concat([df_encoded, missing_data], axis=1)

        # Ensure the column order matches what the model was trained with
        df_encoded = df_encoded[trained_features]

        # Convert to NumPy array for model prediction
        X_test_data = df_encoded.to_numpy()

        # Apply PCA transformation (same transformation as during training)
        new_data_transformed = pca.transform(X_test_data)

        # Make prediction using the trained RandomForest model
        prediction = model.predict(new_data_transformed)

        # Return the prediction result
        return jsonify({'Flagged_As_laundering': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Return error message if any exception occurs

# Kafka producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)
topic_name = "transactions"

@app.route('/transaction', methods=['POST'])
def transaction():
    """
    Accept a transaction from the frontend,
    then push it to Kafka (the 'transactions' topic).
    """
    try:
        data = request.get_json()
        producer.send(topic_name, data)
        producer.flush()
        return jsonify({'status': 'ok', 'message': 'Transaction sent to Kafka'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
