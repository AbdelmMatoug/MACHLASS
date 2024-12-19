import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Get the last time-step output from the LSTM
        last_lstm_output = lstm_out[:, -1, :]
        output = self.fc(last_lstm_output)
        return output

# Function to preprocess the data (scaling)
def preprocess_data(df, feature_columns):
    df = df[feature_columns]
    
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    
    # Scale the features
    df_scaled = scaler.fit_transform(df.values)
    
    return df_scaled, scaler

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    X = []
    y = []
    
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])  # Predict the next day's 'Open' price
    
    return np.array(X), np.array(y)

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        y_pred = model(X_test)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    y_pred_np = y_pred.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    mape = mean_absolute_percentage_error(y_test_np, y_pred_np)
    
    return mape, y_pred

# Main function
def main(args):
    """ Main entry point of the app """
    train_file = args.training_file
    test_file = args.testing_file
    
    # Load train and test datasets
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    # Define feature columns based on your dataset
    feature_columns = ['Open', 'High', 'Low', 'Last Close']  # Corrected feature column names
    
    # Preprocess training and testing data
    train_scaled, scaler = preprocess_data(train_data, feature_columns)
    test_scaled, _ = preprocess_data(test_data, feature_columns)

    # Create sequences for LSTM input
    seq_length = 60  # Number of previous days to consider for prediction
    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_test, y_test = create_sequences(test_scaled, seq_length)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Initialize the LSTM model
    model = LSTMModel(input_size=X_train.shape[2], hidden_layer_size=50, num_layers=2, output_size=1)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    epochs = 10
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        optimizer.zero_grad()
        
        y_pred_train = model(X_train_tensor)
        loss = criterion(y_pred_train, y_train_tensor)
        
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    # Evaluate the model on the test data
    mape, y_pred_test = evaluate_model(model, X_test_tensor, y_test_tensor)
    
    print(f"MAPE on the test set: {mape}")

    # Optionally: inverse transform predictions to original scale
    y_pred_test_original = scaler.inverse_transform(y_pred_test.cpu().numpy())
    y_test_original = scaler.inverse_transform(y_test_tensor.cpu().numpy())

    print("Test predictions: ", y_pred_test_original)
    print("Actual values: ", y_test_original)

# If running as a script, process arguments and execute the main function
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('training_file', type=str, help='Path to the training data CSV')
    parser.add_argument('testing_file', type=str, help='Path to the testing data CSV')
    
    args = parser.parse_args()
    
    main(args)
