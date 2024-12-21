import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from itertools import product
from tqdm import tqdm
import random
import joblib

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
# Load data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
        data = data.sort_values(by='Date')
        return data
    except Exception as e:
        raise ValueError(f"Error loading {file_path}: {e}")

# Preprocess data
def preprocess_data(data, feature_cols, target_col, sequence_length=60):
    if not all(col in data.columns for col in feature_cols + [target_col]):
        raise ValueError("Missing required columns in the dataset.")

    data = data[feature_cols + [target_col]].dropna()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, :-1])
        y.append(scaled_data[i, -1])

    X = np.array(X)
    y = np.array(y)

    # Save scaler for reproducibility
    joblib.dump(scaler, 'scaler.pkl')

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), scaler

# Train model
def train_model(model, X_train, y_train, num_epochs=100, learning_rate=0.001, batch_size=32, shuffle=True, seed=42):
    set_seed(seed)  # Reset seed for reproducibility
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(model.fc.weight.device), batch_y.to(model.fc.weight.device)

            optimizer.zero_grad()
            outputs = model(batch_X).squeeze(-1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

# Predict with continuation
def predict_with_continuation(model, X_train, test_data_length, scaler, feature_cols):
    model.eval()
    with torch.no_grad():
        predictions = []
        current_input = X_train[-1].unsqueeze(0)

        for _ in range(test_data_length):
            prediction = model(current_input).squeeze(-1)
            predictions.append(prediction.item())

            next_input = torch.zeros_like(current_input)
            next_input[0, :-1, :] = current_input[0, 1:, :]
            next_input[0, -1, :] = prediction.unsqueeze(-1)
            current_input = next_input

    predictions = np.array(predictions).reshape(-1, 1)
    inv_predictions = np.zeros((len(predictions), len(feature_cols) + 1))
    inv_predictions[:, -1] = predictions[:, 0]
    inv_predictions = scaler.inverse_transform(inv_predictions)

    return inv_predictions[:, -1]

# Plot predictions
def plot_predictions(test_dates, test_actual, test_predicted, filename, param_dict, mape):
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, test_actual, label='Test Actual', marker='o', color='blue')
    plt.plot(test_dates, test_predicted, label='Test Predicted', marker='x', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f"Hidden: {param_dict['hidden_size']}, Layers: {param_dict['num_layers']}, LR: {param_dict['learning_rate']:.5f}, Batch: {param_dict['batch_size']}, Epochs: {param_dict['num_epochs']}, SeqLen: {param_dict['sequence_length']}, MAPE: {mape:.2f}%")
    plt.legend()
    plt.grid(True)
    try:
        plt.savefig(filename)
        print(f"Plot saved as {filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close()

# Calculate MAPE
def calculate_mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100

def grid_search(args, parameter_grid, max_runs=None):
    set_seed(42)  # Reset seed
    train_data = load_data(args.training_file)
    test_data = load_data(args.testing_file)

    feature_cols = ['Open', 'High', 'Low']
    target_col = 'Last Close'

    sequence_length = max(parameter_grid.get('sequence_length', [1]))
    X_train, y_train, scaler = preprocess_data(train_data, feature_cols, target_col, sequence_length)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_mape = float('inf')
    best_params = None

    param_combinations = list(product(*parameter_grid.values()))
    if max_runs and max_runs < len(param_combinations):
        param_combinations = random.sample(param_combinations, max_runs)
    total_runs = len(param_combinations)

    with tqdm(total=total_runs, desc="Grid Search Progress") as pbar:
        for idx, params in enumerate(param_combinations):
            set_seed(42)
            param_dict = dict(zip(parameter_grid.keys(), params))
            print(f"Run {idx + 1}/{total_runs}, Testing parameters: {param_dict}")

            model = LSTMModel(
                input_size=len(feature_cols),
                hidden_size=param_dict['hidden_size'],
                num_layers=param_dict['num_layers'],
                output_size=1
            ).to(device)

            X_train, y_train = X_train.to(device), y_train.to(device)

            try:
                train_model(
                    model,
                    X_train,
                    y_train,
                    num_epochs=param_dict['num_epochs'],
                    learning_rate=param_dict['learning_rate'],
                    batch_size=param_dict['batch_size'],
                    shuffle=False
                )

                test_predictions = predict_with_continuation(model, X_train, len(test_data), scaler, feature_cols)
                test_actual = test_data['Last Close'].iloc[:len(test_predictions)].values
                mape = calculate_mape(test_actual, test_predictions)

                print(f"MAPE for parameters {param_dict}: {mape:.2f}%")

                if mape < best_mape:
                    best_mape = mape
                    best_params = param_dict
                    torch.save(model.state_dict(), 'best_model.pth')  # Save best model state
                    print(f"New best parameters found with MAPE: {best_mape:.2f}%")

            except Exception as e:
                print(f"Error with parameters {param_dict}: {e}")

            pbar.update(1)

    print(f"Best parameters: {best_params}")
    print(f"Best MAPE: {best_mape:.2f}%")
    return best_params


def main(args):
    set_seed(42)

    # Check if a pre-trained model exists
    if torch.cuda.is_available() and torch.backends.cudnn.version() and os.path.exists('best_model.pth'):
        print("Using saved model for prediction.")
        
        train_data = load_data(args.training_file)
        test_data = load_data(args.testing_file)

        feature_cols = ['Open', 'High', 'Low']
        target_col = 'Last Close'

        # Reload the scaler
        scaler = joblib.load('scaler.pkl')
        sequence_length = 20  # Ensure this matches the sequence length used during training
        
        # Preprocess test data
        X_train, y_train, _ = preprocess_data(train_data, feature_cols, target_col, sequence_length)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the model
        model = LSTMModel(
            input_size=len(feature_cols),
            hidden_size=32,  # Match the best parameters from saved training
            num_layers=2,  # Match the best parameters from saved training
            output_size=1
        ).to(device)

        # Load saved weights
        model.load_state_dict(torch.load('best_model.pth'))
        model.eval()  # Set to evaluation mode

        # Generate predictions
        test_predictions = predict_with_continuation(model, X_train, len(test_data), scaler, feature_cols)
        test_actual = test_data['Last Close'].iloc[:len(test_predictions)].values
        mape = calculate_mape(test_actual, test_predictions)

        print(f"Final Model MAPE on Test Data: {mape:.2f}%")

        # Plot predictions
        final_plot_filename = f"final_hidden32_layers2_lr0.005_batch64_epochs50_seq20_mape{mape:.2f}.png"
        plot_predictions(test_data['Date'], test_actual, test_predictions, final_plot_filename, {
            'hidden_size': 32,
            'num_layers': 2,
            'learning_rate': 0.005,
            'batch_size': 64,
            'num_epochs': 50,
            'sequence_length': 20
        }, mape)
    else:
        print("No pre-trained model found. Running grid search and training.")
        parameter_grid = {
            'hidden_size': [32, 64],
            'num_layers': [1, 2],
            'learning_rate': [0.001, 0.005],
            'batch_size': [32, 64],
            'num_epochs': [50],
            'sequence_length': [20]
        }

        try:
            best_params = grid_search(args, parameter_grid)

            print(f"Best parameters: {best_params}")
        except Exception as e:
            print(f"An error occurred during grid search or final testing: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("training_file", help="Training data file")
    parser.add_argument("testing_file", help="Testing data file")
    args = parser.parse_args()
    main(args)