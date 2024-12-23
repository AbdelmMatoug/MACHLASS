from tqdm import tqdm
import sys
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_percentage_error
from itertools import product
from tqdm import tqdm


class Trainer:
    @staticmethod
    def train_model(
        model,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        num_epochs=100,
        learning_rate=0.001,
        batch_size=32,
        combination_idx=None,  # Index of the current combination in grid search
        total_combinations=None  # Total number of grid search combinations
    ):
        """
        Train the given model with specified parameters.

        Args:
            model: The PyTorch model to train.
            X_train, y_train: Training data.
            X_val, y_val: Validation data (optional).
            num_epochs: Number of training epochs.
            learning_rate: Learning rate for the optimizer.
            batch_size: Batch size for training.
            combination_idx: Index of the current grid search combination.
            total_combinations: Total number of grid search combinations.
        """
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        best_val_loss = float("inf")
        epoch_times = []  # Track the duration of each epoch

        # Print the combination progress if indices are provided
        if combination_idx is not None and total_combinations is not None:
            print(f"Combination {combination_idx}/{total_combinations}: Starting training...")

        # Loop through all the epochs
        for epoch in range(num_epochs):
            start_time = time.time()
            model.train()
            train_loss = 0.0

            # Training loop over batches
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(model.fc.weight.device), batch_y.to(model.fc.weight.device)

                optimizer.zero_grad()
                outputs = model(batch_X).squeeze(-1)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Compute average training loss
            train_loss /= len(dataloader)

            # Validation
            val_loss, val_mape = None, None
            if X_val is not None and y_val is not None:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val.to(model.fc.weight.device)).squeeze(-1).cpu().numpy()
                    val_loss = criterion(
                        torch.tensor(val_outputs, dtype=torch.float32),
                        y_val.to(model.fc.weight.device),
                    ).item()
                    val_actual = y_val.cpu().numpy()
                    val_mape = mean_absolute_percentage_error(val_actual, val_outputs)

            # Handle val_loss and formatting correctly
            val_loss_str = f"{val_loss:.4f}" if val_loss is not None else 'N/A'

            # Display epoch progress
            sys.stdout.write(
                f"\rEpoch [{epoch + 1}/{num_epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss_str} | "
                f"Val MAPE: {val_mape * 100:.2f}%"
            )
            sys.stdout.flush()

            epoch_time = time.time() - start_time
            epoch_times.append(epoch_time)

            avg_time_per_epoch = sum(epoch_times) / len(epoch_times)
            remaining_epochs = num_epochs - (epoch + 1)
            est_time_remaining = avg_time_per_epoch * remaining_epochs

            # Display estimated time remaining
            sys.stdout.write(
                f" | Time/Epoch: {epoch_time:.2f}s | "
                f"ETA: {est_time_remaining:.2f}s"
            )
            sys.stdout.flush()

        print("\nTraining complete.")
