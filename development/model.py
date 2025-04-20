import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

# 1. Load and preprocess data
data = pd.read_csv("training_data/preprocessed_data.csv")

# Separate non-numerical columns (like timestamp)
non_numerical_columns = ["timestamp", "longitude", "latitude"]
numerical_columns = [col for col in data.columns if col not in non_numerical_columns]

# Normalize only numerical columns
scaler = MinMaxScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

scaler_file = "scaler.pkl"
joblib.dump(scaler, scaler_file)
print(f"Scaler saved to {scaler_file}.")

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

# Convert to PyTorch datasets
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 2. Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

# 3. Hyperparameter Tuning using Random Search
def random_search_lstm(hyperparams, X_train, y_train, X_test, y_test):
    hidden_dims = hyperparams['hidden_dim']
    num_layers = hyperparams['num_layers']
    learning_rates = hyperparams['learning_rate']
    
    best_model = None
    best_mse = float('inf')
    best_params = None

    for hidden_dim, num_layer, lr in itertools.product(hidden_dims, num_layers, learning_rates):
        print(f"Testing: hidden_dim={hidden_dim}, num_layers={num_layer}, learning_rate={lr}")
        model = LSTMModel(input_dim=X_train.shape[2], hidden_dim=hidden_dim, output_dim=1, num_layers=num_layer)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Train the model
        for epoch in range(EPOCHS):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                loss.backward()
                optimizer.step()

        # Evaluate the model
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                y_pred.extend(outputs.squeeze().tolist())
                y_true.extend(y_batch.tolist())

        mse = mean_squared_error(y_true, y_pred)
        print(f"Validation MSE: {mse:.4f}")

        # Track the best model
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_params = {'hidden_dim': hidden_dim, 'num_layers': num_layer, 'learning_rate': lr}
    
    return best_model, best_params, best_mse

# Define hyperparameter ranges for random search
hyperparams = {
    'hidden_dim': [32, 64, 128],
    'num_layers': [1, 2, 3],
    'learning_rate': [0.001, 0.01, 0.1]
}

# Run random search
best_model, best_params, best_mse = random_search_lstm(hyperparams, X_train, y_train, X_test, y_test)

print(f"Best Model Parameters: {best_params}")
print(f"Best Validation MSE: {best_mse:.4f}")

# Corrected input example with 9 features
input_example = pd.DataFrame({
    "co": [1295.09],
    "no": [3.1],
    "no2": [41.81],
    "o3": [100.14],
    "so2": [6.5],
    "pm2_5": [78.0],
    "pm10": [96.69],
    "nh3": [11.15],
    "extra_feature": [0.0]  # Replace with correct feature
})

mlflow.set_experiment("Time-Series Air Quality Forecasting")

with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metric("best_mse", best_mse)
    
    # Log the model with corrected input example
    mlflow.pytorch.log_model(
        best_model, 
        "best_time_series_model", 
        input_example=input_example
    )

    # Save preprocessed data
    mlflow.log_artifact("training_data/preprocessed_data.csv")

    print("Best model and scaler logged to MLflow with corrected input example.")
    
print("Hyperparameter tuning complete. Best model logged to MLflow.")
