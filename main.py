import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import logging
import random
import joblib
from itertools import product

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Seed for reproducibility
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

# Load and preprocess data
def load_data(file_path, sequence_length=60, feature_cols=None, target_col=None, split_ratio=0.7):
    logging.info(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
    data = data.sort_values(by='Date')
    logging.info(f"Data loaded. Total records: {len(data)}")

    if not all(col in data.columns for col in feature_cols + [target_col]):
        raise ValueError("Missing required columns in the dataset.")

    dates = data['Date']
    data = data[feature_cols + [target_col]].dropna()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, :-1])
        y.append(scaled_data[i, -1])

    X, y = np.array(X), np.array(y)

    split_idx = int(len(X) * split_ratio)
    if split_idx < 1 or len(X) - split_idx < 1:
        raise ValueError("Split ratio results in insufficient training or validation data.")

    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    logging.info(f"Training data date range: {dates.iloc[:split_idx].min()} to {dates.iloc[:split_idx].max()}")
    logging.info(f"Validation data date range: {dates.iloc[split_idx:].min()} to {dates.iloc[split_idx:].max()}")
    logging.info(f"Time-based split: {len(X_train)} training samples, {len(X_val)} validation samples.")
    logging.info(f"Training data scaled range: {scaled_data[:split_idx].min(axis=0)} to {scaled_data[:split_idx].max(axis=0)}")
    logging.info(f"Validation data scaled range: {scaled_data[split_idx:].min(axis=0)} to {scaled_data[split_idx:].max(axis=0)}")
    logging.debug(f"Data Head:\n{data.head()}")
    logging.debug(f"Feature Scaling Ranges (Training): Min {scaled_data[:split_idx].min(axis=0)}, Max {scaled_data[:split_idx].max(axis=0)}")
    logging.debug(f"Feature Scaling Ranges (Validation): Min {scaled_data[split_idx:].min(axis=0)}, Max {scaled_data[split_idx:].max(axis=0)}")

    joblib.dump(scaler, 'scaler.pkl')
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
        scaler
    )

def train_model(model, X_train, y_train, X_val=None, y_val=None, num_epochs=100, learning_rate=0.001, batch_size=32):
    logging.info("Starting training...")
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    config = {
        'input_size': model.lstm.input_size,
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers,
        'output_size': model.fc.out_features
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for i, (batch_X, batch_y) in enumerate(dataloader):
            batch_X, batch_y = batch_X.to(model.fc.weight.device), batch_y.to(model.fc.weight.device)

            optimizer.zero_grad()
            outputs = model(batch_X).squeeze(-1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if (i + 1) % 10 == 0:  # Log every 10 batches
                logging.debug(f"Epoch {epoch + 1}, Batch {i + 1}/{len(dataloader)}, Batch Loss: {loss.item():.4f}")

        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val.to(model.fc.weight.device)).squeeze(-1).cpu().numpy()
                val_loss = criterion(torch.tensor(val_outputs), y_val.to(model.fc.weight.device)).item()
                val_actual = y_val.cpu().numpy()
                mape = mean_absolute_percentage_error(val_actual, val_outputs)
            if epoch % 10 == 0 or epoch == num_epochs - 1:  # Log only every 10th epoch or the final epoch

                logging.info(
                    f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss / len(dataloader):.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val MAPE: {mape:.4f}"
                )

            if val_loss < best_val_loss:
                logging.debug(f"New Best Validation Loss: {val_loss:.4f}")
                best_val_loss = val_loss
                torch.save({'model_state_dict': model.state_dict(), 'config': config}, 'best_model.pth')

        else:
            if epoch % 10 == 0 or epoch == num_epochs - 1:  # Log only every 10th epoch or the final epoch
                logging.info(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss / len(dataloader):.4f}")

    logging.info("Training complete.")


from tqdm import tqdm

from tqdm import tqdm
import time

def grid_search(X_train, y_train, X_val, y_val, param_grid, device):
    logging.info("Starting grid search...")
    param_combinations = list(product(*param_grid.values()))
    total_combinations = len(param_combinations)
    best_mape = float('inf')
    best_params = None
    best_checkpoint = None  # To store the best model's checkpoint

    # Initialize the progress bar
    with tqdm(total=total_combinations, desc="Grid Search Progress", unit="combination") as pbar:
        start_time = time.time()  # Track start time

        for idx, param_set in enumerate(param_combinations):
            params = dict(zip(param_grid.keys(), param_set))
            logging.info(f"Testing parameters {idx+1}/{total_combinations}: {params}")

            model = LSTMModel(
                input_size=X_train.shape[2],
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                output_size=1
            ).to(device)

            # Train the model and save the checkpoint
            train_model(
                model, X_train, y_train, X_val, y_val,
                num_epochs=params['num_epochs'],
                learning_rate=params['learning_rate'],
                batch_size=params['batch_size']
            )

            # Load checkpoint for evaluation
            checkpoint = torch.load('best_model.pth', map_location=device)
            config = checkpoint['config']
            logging.info(f"Loaded model configuration: {config}")
            model.load_state_dict(checkpoint['model_state_dict'])  # Load model state
            model.eval()

            # Evaluate the model on validation data
            with torch.no_grad():
                val_predictions = model(X_val.to(device)).squeeze(-1).cpu().numpy()
                val_actual = y_val.cpu().numpy()
                mape = mean_absolute_percentage_error(val_actual, val_predictions)

            logging.info(f"Validation Results | Parameters: {params} | MAPE: {mape:.4f}")
            logging.debug(f"Validation MAPE for parameters {params}: {mape:.4f}")

            if mape < best_mape:
                logging.debug(f"New Best MAPE: {mape:.4f} with parameters {params}")
                best_mape = mape
                best_params = params
                best_checkpoint = checkpoint  # Save the best checkpoint

            # Update progress bar
            elapsed_time = time.time() - start_time
            avg_time_per_combination = elapsed_time / (idx + 1)
            remaining_time = avg_time_per_combination * (total_combinations - idx - 1)
            pbar.set_postfix({
                "Best MAPE": f"{best_mape:.4f}",
                "ETA": f"{remaining_time:.2f}s"
            })
            pbar.update(1)

    # Save the best checkpoint to use later
    torch.save(best_checkpoint, 'best_model_final.pth')
    logging.info(f"Best parameters: {best_params} | Best MAPE: {best_mape:.4f}")
    return best_params


def predict_future(model, input_data, prediction_steps, scaler, feature_cols):
    logging.info("Starting future predictions...")
    model.eval()
    predictions = []
    current_input = input_data.clone()

    with torch.no_grad():
        for step in range(prediction_steps):
            prediction = model(current_input.unsqueeze(0)).squeeze(-1)
            predictions.append(prediction.item())

            next_input = torch.zeros_like(current_input)
            next_input[:-1, :] = current_input[1:, :]
            next_input[-1, :] = torch.tensor(prediction.item(), dtype=torch.float32)
            current_input = next_input

            if (step + 1) % 5 == 0:
                logging.info(f"Step {step + 1} | Predicted: {prediction.item():.4f} | Input: {current_input[-1].cpu().numpy()}")
  

    predictions = np.array(predictions).reshape(-1, 1)
    inv_predictions = np.zeros((len(predictions), len(feature_cols) + 1))
    inv_predictions[:, -1] = predictions[:, 0]
    inv_predictions = scaler.inverse_transform(inv_predictions)

    logging.info(f"Final Scaled Predictions Range: Min {predictions.min():.4f}, Max {predictions.max():.4f}")
    return inv_predictions[:, -1]

def plot_predictions(dates, actual, predicted, title, filename):
    logging.info("Plotting predictions...")
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual', marker='o', color='blue')
    plt.plot(dates, predicted, label='Predicted', marker='x', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    logging.info(f"Plot saved as {filename}")

# Main function
import os

def main(args):
    set_seed(42)
    logging.info("Starting the process...")

    feature_cols = ['Open', 'High', 'Low', 'Last Close']
    target_col = 'Last Close'
    logging.debug(f"Feature Columns: {feature_cols}, Target Column: {target_col}")

    logging.info("Loading and preprocessing training data...")
    X_train, y_train, X_val, y_val, scaler = load_data(
        args.training_file, feature_cols=feature_cols, target_col=target_col
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check if a saved model exists
    model_path = 'best_model_final.pth'
    if os.path.exists(model_path):
        response = input("A saved model exists. Do you want to train a new model? (yes/no): ").strip().lower()
    else:
        response = "yes"  # If no saved model exists, force training

    if response == "yes":
        logging.info("Starting model training...")
        param_grid = {
            'hidden_size': [32, 64, 128],
            'num_layers': [1, 2, 3,4, 5],
            'learning_rate': [0.01,0.001, 0.0001, 0.005],
            'batch_size': [16, 32, 64],
            'num_epochs': [50, 100,150]
        }
        best_params = grid_search(X_train, y_train, X_val, y_val, param_grid, device)
        logging.debug(f"Grid Search Parameters: {param_grid}")
        logging.debug(f"Best Parameters: {best_params}")
        checkpoint = torch.load('best_model_final.pth', map_location=device)
        config = checkpoint['config']
    else:
        logging.info(f"Loading existing model from {model_path}...")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            config = checkpoint['config']
        except FileNotFoundError:
            logging.error("No saved model found. Please train a model first.")
            return
        except KeyError as e:
            logging.error(f"Missing key in checkpoint: {e}")
            return
        except RuntimeError as e:
            logging.error(f"Error loading the model state_dict: {e}")
            return

    model = LSTMModel(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_size=config['output_size']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info("Model loaded successfully.")

    logging.info("Loading and processing test data...")
    test_data = pd.read_csv(args.testing_file)
    test_dates = pd.to_datetime(test_data['Date'], format='%m/%d/%Y')
    test_predictions = predict_future(
        model, X_train[-1, :, :], len(test_data), scaler, feature_cols
    )

    logging.info("Predictions complete. Plotting results...")
    test_actual = test_data[target_col].values
    plot_predictions(
        test_dates, test_actual, test_predictions,
        title=f"Best Model Predictions", filename="best_model_predictions.png"
    )
    test_scaled_data = scaler.transform(test_data[feature_cols + [target_col]].dropna())
    logging.info(f"Test data scaled range: {test_scaled_data.min(axis=0)} to {test_scaled_data.max(axis=0)}")

    mape = mean_absolute_percentage_error(test_actual, test_predictions)
    print(test_predictions)
    logging.info(f"Final Test MAPE: {mape:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("training_file", help="Training data file")
    parser.add_argument("testing_file", help="Testing data file")
    args = parser.parse_args()
    main(args)