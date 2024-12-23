from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_percentage_error
import torch
import random
import time
import sys
from itertools import product
from model import LSTMModel
import numpy as np
from plot_utils import plot_training_validation_loss

def set_seed(seed):
    """
    Set the random seed for reproducibility across multiple runs.
    
    This ensures consistent results by fixing the random number generator's seed
    for Python, NumPy, and PyTorch.
    
    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def train_model(model, X_train, y_train, X_val=None, y_val=None, num_epochs=100, learning_rate=0.001, batch_size=32):
    """
    Train the LSTM model on the training dataset.
    
    Args:
        model (nn.Module): The LSTM model to be trained.
        X_train (torch.Tensor): Training input sequences.
        y_train (torch.Tensor): Training target values.
        X_val (torch.Tensor, optional): Validation input sequences.
        y_val (torch.Tensor, optional): Validation target values.
        num_epochs (int): Number of epochs to train the model.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.

    """
    print("Training the model... This may take a while.")
    train_losses = []  # To store training losses
    val_losses = []    # To store validation losses
    # Wrap training data in a PyTorch DataLoader for batch processing
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Define the loss function (Mean Squared Error) and optimizer (Adam)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Variables to track the best validation loss for model checkpointing
    best_val_loss = float('inf')
    config = {
        'input_size': model.lstm.input_size,
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers,
        'output_size': model.fc.out_features
    }

    epoch_times = []  # Track duration of each epoch

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()  # Set the model to training mode
        train_loss = 0.0
     

        # Training loop: Process batches
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(model.fc.weight.device), batch_y.to(model.fc.weight.device)

            optimizer.zero_grad()  # Clear gradients from the previous step
            outputs = model(batch_X).squeeze(-1)  # Forward pass
            loss = criterion(outputs, batch_y)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            train_loss += loss.item()  # Accumulate training loss

        train_loss /= len(dataloader)  # Average training loss for the epoch
        train_losses.append(train_loss)
        # Validation loop: Evaluate on validation data if available
        val_loss, val_mape = None, None
        if X_val is not None and y_val is not None:
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                val_outputs = model(X_val.to(model.fc.weight.device)).squeeze(-1).cpu().numpy()
                val_loss = criterion(
                    torch.tensor(val_outputs, dtype=torch.float32),
                    y_val.to(model.fc.weight.device)
                ).item()
                val_actual = y_val.cpu().numpy()
                val_mape = mean_absolute_percentage_error(val_actual, val_outputs)
                val_losses.append(val_loss)
            # Save the model if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({'model_state_dict': model.state_dict(), 'config': config}, 'best_model.pth')

        # Calculate time taken for the epoch
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        # Estimate time remaining
        avg_time_per_epoch = sum(epoch_times) / len(epoch_times)
        remaining_epochs = num_epochs - (epoch + 1)
        est_time_remaining = avg_time_per_epoch * remaining_epochs

        # Dynamically print progress for the current epoch
        sys.stdout.write(
            f"\rEpoch [{epoch + 1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val MAPE: {val_mape * 100:.2f}% | "
            f"Time/Epoch: {epoch_time:.2f}s | "
            f"ETA: {est_time_remaining:.2f}s"
        )
        sys.stdout.flush()
    plot_training_validation_loss(
        train_losses, val_losses,
        "Training and Validation Loss Over Epochs",
        "training_validation_loss.png"
    )
    print("\nTraining complete.")


def perform_grid_search(param_grid, X_train, y_train, X_val, y_val, input_size, scaler):
    """
    Perform a grid search to find the best hyperparameters for the LSTM model.
    
    Args:
        param_grid (dict): Dictionary of hyperparameters to test.
        X_train (torch.Tensor): Training input sequences.
        y_train (torch.Tensor): Training target values.
        X_val (torch.Tensor): Validation input sequences.
        y_val (torch.Tensor): Validation target values.
        input_size (int): Number of features in the input data.
        scaler (MinMaxScaler): Scaler used for data normalization.

    Returns:
        best_model (nn.Module): The model with the best hyperparameters.
        best_params (dict): The best hyperparameters found during grid search.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_params = None
    best_model = None
    best_val_loss = float('inf')
    criterion = torch.nn.MSELoss()

    param_combinations = list(product(*param_grid.values()))  # Generate all parameter combinations
    total_combinations = len(param_combinations)
    print(f"Starting grid search with {total_combinations} combinations...")

    # Iterate through all combinations of parameters
    for idx, params in enumerate(param_combinations, start=1):
        param_dict = dict(zip(param_grid.keys(), params))
        print(f"\n[{idx}/{total_combinations}] Testing parameters: {param_dict}")

        model = LSTMModel(
            input_size=input_size,
            hidden_size=param_dict['hidden_size'],
            num_layers=param_dict['num_layers'],
            output_size=1
        ).to(device)

        # Train the model with the current parameter set
        train_model(
            model, X_train, y_train, X_val, y_val,
            num_epochs=param_dict['num_epochs'],
            learning_rate=param_dict['learning_rate'],
            batch_size=param_dict['batch_size']
        )

        # Load the best model from the saved checkpoint
        checkpoint = torch.load('best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate on validation data
        val_loss, val_mape = None, None
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val.to(model.fc.weight.device)).squeeze(-1).cpu().numpy()
                val_loss = criterion(
                    torch.tensor(val_outputs, dtype=torch.float32),
                    y_val.to(model.fc.weight.device)
                ).item()
                val_actual = y_val.cpu().numpy()
                val_mape = mean_absolute_percentage_error(val_actual, val_outputs)

        # Update the best model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = param_dict
            best_model = model

        print(f"[{idx}/{total_combinations}] Val Loss: {val_loss:.4f}, Val MAPE: {val_mape * 100:.2f}%")

    print(f"\nGrid search complete. Best Parameters: {best_params}, Best Validation Loss: {best_val_loss:.4f}")
    return best_model, best_params


def predict_future(model, input_data, prediction_steps, scaler, feature_cols):
    """
    Predict future values based on the trained model.

    Args:
        model (nn.Module): Trained LSTM model.
        input_data (torch.Tensor): Input sequence for prediction.
        prediction_steps (int): Number of time steps to predict into the future.
        scaler (MinMaxScaler): Scaler used for data normalization.
        feature_cols (list): List of feature column names.

    Returns:
        np.ndarray: Unscaled predicted values.
    """
    print("Generating future predictions...")
    model.eval()
    predictions = []
    current_input = input_data.clone()

    with torch.no_grad():
        for step in range(prediction_steps):
            # Predict the next value
            prediction = model(current_input.unsqueeze(0)).squeeze(-1)
            predictions.append(prediction.item())

            # Update the input sequence for the next prediction
            next_input = torch.zeros_like(current_input)
            next_input[:-1, :] = current_input[1:, :]  # Shift previous inputs
            next_input[-1, :] = prediction  # Add the predicted value
            current_input = next_input

    # Transform predictions back to the original scale
    predictions = np.array(predictions).reshape(-1, 1)
    inv_predictions = np.zeros((len(predictions), len(feature_cols) + 1))
    inv_predictions[:, -1] = predictions[:, 0]
    inv_predictions = scaler.inverse_transform(inv_predictions)

    return inv_predictions[:, -1]
