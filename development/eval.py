import mlflow.pytorch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_squared_error
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import mlflow
import mlflow.pytorch
import itertools
from sklearn.metrics import mean_squared_error
import joblib

# Configurations
SEQUENCE_LENGTH = 10  # Length of input sequences for LSTM
TARGET_COLUMN = "air_quality_index"
EPOCHS = 50
BATCH_SIZE = 64

# Load the best model from MLflow
model_uri = "mlruns/878626530893598251/fa4fbbdc580c48e696572d3607f7b9ab/artifacts/best_time_series_model"  # Replace with the actual URI of your model in MLflow
model = mlflow.pytorch.load_model(model_uri)
data = pd.read_csv("training_data/preprocessed_data.csv")

# Separate non-numerical columns (like timestamp)
non_numerical_columns = ["timestamp", "longitude", "latitude"]
numerical_columns = [col for col in data.columns if col not in non_numerical_columns]

# Normalize only numerical columns
scaler = MinMaxScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])



# Convert DataFrame to time-series sequences
def create_sequences(data, target_column, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :])
        y.append(data[i + seq_length, target_column])
    return np.array(X), np.array(y)

# Use only numerical data for modeling
numerical_data = data[numerical_columns].to_numpy()
target_index = numerical_columns.index(TARGET_COLUMN)
X, y = create_sequences(numerical_data, target_index, SEQUENCE_LENGTH)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Set the model to evaluation mode
model.eval()

# Predict on test data
with torch.no_grad():
    outputs = model(X_test_tensor)
    # Convert continuous predictions to binary labels
    predictions = outputs.squeeze().round()
    mse = mean_squared_error(y_test, predictions.numpy())
    print(f"Mean Squared Error: {mse:.4f}")

# Convert y_test to binary labels
y_test_binary = (y_test >= 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test_binary, predictions.numpy())
print(f"Confusion Matrix:\n{cm}")

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
