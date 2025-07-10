import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from collections import defaultdict

# Load the saved models and preprocessing objects
rf = joblib.load('random_forest_model.pkl')
pca = joblib.load('pca_model.pkl')
trained_features = joblib.load('trained_features.pkl')
encoder = joblib.load('encoder.pkl')

# Define the categorical columns used during preprocessing
categorical_columns = [
    "Time", "Date", "Sender_account", "Receiver_account", "Amount",
    "Payment_currency", "Received_currency", "Sender_bank_location",
    "Receiver_bank_location", "Payment_type", "Is_laundering", "Laundering_type"
]

# Initialize accumulators for computing accuracy incrementally and storing predictions
total_correct = 0
total_samples = 0

confusion_counts = defaultdict(lambda: defaultdict(int))
all_y_true = []
all_y_pred = []

# Define the chunk size and CSV file
chunksize = 1000
csv_file = "SAML-D_shuffled.csv"

# Process the CSV file in chunks
for chunk in pd.read_csv(csv_file, dtype={"Sender_account": "str", "Receiver_bank_location": "str"},
                         chunksize=chunksize, low_memory=True):
    # Shuffle the chunk (optional)
    #chunk = chunk.sample(frac=1, random_state=42).reset_index(drop=True)

    # --- Step 1: Process High-Cardinality Columns ---
    # Convert to string and fill missing values before encoding
    high_cardinality_cols = ["Sender_account", "Receiver_account"]
    chunk[high_cardinality_cols] = chunk[high_cardinality_cols].astype(str).fillna("Unknown")
    # Apply the ordinal encoder to these columns
    chunk[high_cardinality_cols] = encoder.transform(chunk[high_cardinality_cols])

    # --- Step 2: One-Hot Encode the Categorical Columns ---
    # Perform one-hot encoding on the specified categorical columns
    chunk_encoded = pd.get_dummies(chunk, columns=categorical_columns, dtype="int8")

    # --- Step 3: Ensure the Feature Set Matches the Training Data ---
    missing_cols = [col for col in trained_features if col not in chunk_encoded.columns]
    if missing_cols:
        missing_data = pd.DataFrame(0, index=chunk_encoded.index, columns=missing_cols)
        chunk_encoded = pd.concat([chunk_encoded, missing_data], axis=1)
    # Reorder the columns to match the training set
    chunk_encoded = chunk_encoded[trained_features]

    # --- Step 4: Recreate the Target Variable ---
    if 'Is_laundering_1' in chunk_encoded.columns:
        chunk_encoded['Is_laundering'] = chunk_encoded['Is_laundering_1']
        chunk_encoded.drop(columns=['Is_laundering_0', 'Is_laundering_1'], inplace=True, errors="ignore")
    else:
        chunk_encoded['Is_laundering'] = chunk['Is_laundering']

    # --- Step 5: Separate Features and Target ---
    X_chunk = chunk_encoded.drop(columns=['Is_laundering'])
    y_chunk = chunk_encoded['Is_laundering']

    # Convert features to a sparse matrix for efficiency
    X_chunk_sparse = csr_matrix(X_chunk.values)

    # Apply the saved PCA transformation
    X_chunk_pca = pca.transform(X_chunk_sparse)

    # --- Step 6: Make Predictions Using the Default predict() Method ---
    y_pred_chunk = rf.predict(X_chunk_pca)

    # Update overall accuracy counts
    correct = np.sum(y_pred_chunk == y_chunk.to_numpy())
    total_correct += correct
    total_samples += len(y_chunk)

    # Accumulate predictions for the full classification report
    all_y_true.extend(y_chunk.tolist())
    all_y_pred.extend(y_pred_chunk.tolist())

    # Update confusion counts for a simple report
    for true_val, pred_val in zip(y_chunk.tolist(), y_pred_chunk.tolist()):
        confusion_counts[true_val][pred_val] += 1

# Compute overall accuracy
accuracy_full = total_correct / total_samples
print("Full dataset accuracy:", accuracy_full)

# Print confusion counts
print("Confusion counts:")
for true_class in sorted(confusion_counts.keys()):
    print(f"True {true_class}: {dict(confusion_counts[true_class])}")

# Print detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(all_y_true, all_y_pred))

# Print the confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(all_y_true, all_y_pred))
