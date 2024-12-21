import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

# Load Data
def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file, parse_dates=['Date'])
    test_data = pd.read_csv(test_file, parse_dates=['Date'])
    return train_data, test_data

# Preprocessing
def preprocess_data(data, scaler=None):
    features = ['Open', 'High', 'Low', 'Last Close']
    if not scaler:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data[features] = scaler.fit_transform(data[features])
    else:
        data[features] = scaler.transform(data[features])
    return data, scaler

# Prepare sequences for LSTM
def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i + seq_length][['Open', 'High', 'Low', 'Last Close']].values
        label = data.iloc[i + seq_length]['Last Close']
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last output of the LSTM
        return out

# Training loop
def train_model(model, train_loader, criterion, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs.squeeze(), y)  # Squeeze predictions
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

# Main script
def main(train_file, test_file):
    train_data, test_data = load_data(train_file, test_file)
    
    # Preprocessing
    train_data, scaler = preprocess_data(train_data)
    test_data, _ = preprocess_data(test_data, scaler=scaler)
    
    # Sequence preparation
    seq_length = 20
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)
    
    # Convert to PyTorch tensors
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)
    
    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), batch_size=32, shuffle=True)
    
    # Model setup
    model = LSTMModel(input_dim=4, hidden_dim=50, output_dim=1, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    train_model(model, train_loader, criterion, optimizer, epochs=50)
    
    # Predict and evaluate
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).squeeze().numpy()  # Squeeze predictions
        y_test_np = y_test.numpy()  # Convert to numpy array for evaluation.

    mape = mean_absolute_percentage_error(y_test_np, predictions)
    print(f"MAPE: {mape:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_test_np)), y_test_np, label="Actual Prices", color='blue')
    plt.plot(range(len(predictions)), predictions, label="Predicted Prices", color='red', linestyle='dashed')
    plt.xlabel("Time")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.title("Actual vs Predicted Prices")
    plt.show()

# Run main
if __name__ == "__main__":
    main("train.csv", "test.csv")
