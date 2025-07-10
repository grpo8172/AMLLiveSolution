import pandas as pd
import anndata
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import joblib
import scanpy as sc
from sklearn.metrics import classification_report

# Loading the dataset
df_full = pd.read_csv("SAML-D.csv", dtype={"Sender_account": "str", "Receiver_account": "str"}, low_memory=True)

# Shuffling and then sampling balanced classes
df_shuffled = df_full.sample(frac=1, random_state=42).reset_index(drop=True)

# Applying stratified sampling based on class balance (using minimum samples for each class)
min_samples = df_shuffled['Is_laundering'].value_counts().min()

df_balanced = df_shuffled.groupby('Is_laundering', group_keys=False).apply(
    lambda x: x.sample(n=min(min_samples, 20000), random_state=42) #Not much of a gain from increasing the number of samples as the model is limitted by the minimum number of Is_laundering = 1 cases
).reset_index(drop=True)

# Applying ordinal encoding to high cardinality columns
high_cardinality_cols = ["Sender_account", "Receiver_account"]
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df_balanced[high_cardinality_cols] = encoder.fit_transform(df_balanced[high_cardinality_cols])

# Ensuring 'Is_laundering' is treated as an integer for the target
df_balanced["Is_laundering"] = df_balanced["Is_laundering"].astype(int)

# One-hot encoding categorical columns with reduced memory usage  
categorical_columns = [
    "Time", "Date", "Sender_account", "Receiver_account", "Amount",
    "Payment_currency", "Received_currency", "Sender_bank_location",
    "Receiver_bank_location", "Payment_type", "Is_laundering",
    "Laundering_type"
]

df_encoded = pd.get_dummies(df_balanced, columns=categorical_columns, dtype="int8")

# Dropping highly correlated features like 'Laundering_type'
drop_cols = [col for col in df_encoded.columns if col.startswith("Laundering_type")]
df_encoded.drop(columns=drop_cols, inplace=True, errors="ignore")

# Ensuring no 'object' columns remain
df_encoded = df_encoded.select_dtypes(exclude=['object'])

# Recombining the one-hot encoded columns into a single target column (Is_laundering)
df_encoded['Is_laundering'] = df_encoded['Is_laundering_1']  # Use 'Is_laundering_1' as the target

# Dropping the one-hot encoded columns
df_encoded = df_encoded.drop(columns=['Is_laundering_0', 'Is_laundering_1'])

# Shuffling the dataset to ensure randomness
df_encoded = df_encoded.sample(frac=1, random_state=42).reset_index(drop=True)

# Separating the target column (Is_laundering) and the features
X = df_encoded.drop(columns=['Is_laundering'])  # Features to be created later
y = df_encoded['Is_laundering']  # Target label (laundering or non-laundering)

# Printing to check the result
print(X.head())  # Features to be created later
print(y.head())  # Target (Is_laundering)

# Converting to sparse matrix for efficient memory usage
X_sparse = csr_matrix(X.values)

# Creating AnnData object for compatibility with Scanpy
adata = anndata.AnnData(
    X=X_sparse,  # Assuming X_sparse contains feature data
    obs=df_balanced.drop(columns=['Is_laundering'])
)

# Normalizing and log-transform the data
sc.pp.normalize_total(adata, target_sum=5e3)
sc.pp.log1p(adata)

# Running PCA with reduced components
n_pcs = min(adata.shape[1], 80)
sc.tl.pca(adata, svd_solver='arpack', n_comps=n_pcs)

# Fitting PCA and transform data
pca = PCA(n_components=n_pcs)
X_pca = pca.fit_transform(X_sparse)
adata.obsm['X_pca'] = X_pca  # Store transformed data to use as features

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Checking the distribution of the target label after splitting
print("Train set distribution of Is_laundering:")
print(y_train.value_counts())


# Increase the weight for class 1 (money laundering)
class_weights = {0: 1, 1: 7}  # Adjust these numbers as needed

# Instantiate and train the model
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=30,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight=class_weights,  # Incorporating class weights
    random_state=42
)


# Initializing RandomForestClassifier with class balancing
#rf = RandomForestClassifier(
#    n_estimators=300,  # Increase number of trees for more accuracy
#    max_depth=30,      # Increase depth
#    min_samples_split=10,  # Minimum samples to split a node
#    min_samples_leaf=5,    # Minimum samples to be in a leaf node
#    class_weight='balanced',
#    random_state=42
#)

# Cross-validation setup (StratifiedKFold ensures balanced class distribution in each fold)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []
for train_index, test_index in kf.split(X_pca, y):
    X_train_cv, X_test_cv = X_pca[train_index], X_pca[test_index]
    y_train_cv, y_test_cv = y[train_index], y[test_index]

    rf.fit(X_train_cv, y_train_cv)
    y_pred = rf.predict(X_test_cv)

    accuracy = accuracy_score(y_test_cv, y_pred)
    cv_scores.append(accuracy)

# Average cross-validation score
print(f"Average Cross-Validation Accuracy: {np.mean(cv_scores)}")

# Training Random Forest classifier on the entire training set
rf.fit(X_train, y_train)

# Making predictions on the test set
y_pred = rf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Printing classification report
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(rf, 'random_forest_model.pkl')

# Save the PCA model
joblib.dump(pca, 'pca_model.pkl')

# Save the list of features used in training
trained_features = X.columns.tolist()  # Get the column names of the features
joblib.dump(trained_features, 'trained_features.pkl')

joblib.dump(encoder, 'encoder.pkl')
