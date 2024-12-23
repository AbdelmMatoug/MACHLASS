import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def check_data_leakage(train_data, test_data, unique_col='Date'):
    """
    Check for data leakage between the training and testing datasets.
    
    Data leakage occurs if the same data appears in both training and testing sets,
    which can lead to overly optimistic evaluation results. This function ensures
    that there is no overlap in the specified unique column (e.g., 'Date').
    
    Args:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.
        unique_col (str): Column used to check for overlap (e.g., 'Date').

    Raises:
        ValueError: If any overlap (data leakage) is detected.
    """
    train_values = set(train_data[unique_col])  # Extract unique values from training data
    test_values = set(test_data[unique_col])   # Extract unique values from testing data
    overlap = train_values.intersection(test_values)  # Check for overlap

    if overlap:
        # Raise an error if any overlapping values are found
        raise ValueError(f"Data leakage detected! Overlap in {unique_col}: {overlap}")
    else:
        # Print confirmation if no leakage is found
        print(f"No data leakage detected. Training and testing datasets are distinct.")


def load_data(file_path, sequence_length=20, feature_cols=None, target_col=None, split_ratio=0.9):
    """
    Load, preprocess, and split the dataset into training and validation sets.
    
    This function performs the following:
    1. Reads the data from a CSV file.
    2. Converts the 'Date' column to a datetime format and sorts the data by date.
    3. Extracts the specified feature columns and target column.
    4. Normalizes the data using MinMaxScaler to ensure the model performs well.
    5. Converts the time-series data into overlapping sequences for LSTM input.
    6. Splits the data into training and validation sets based on the split ratio.

    Args:
        file_path (str): Path to the CSV file containing the dataset.
        sequence_length (int): Number of time steps in each input sequence.
        feature_cols (list): List of feature column names to use as input.
        target_col (str): Name of the target column to predict.
        split_ratio (float): Proportion of data to use for training (default: 0.9).

    Returns:
        tuple: A tuple containing:
            - X_train (torch.Tensor): Training input sequences.
            - y_train (torch.Tensor): Training target values.
            - X_val (torch.Tensor): Validation input sequences.
            - y_val (torch.Tensor): Validation target values.
            - scaler (MinMaxScaler): The scaler object used for normalization.
    """
    # Step 1: Read the data from the CSV file
    data = pd.read_csv(file_path)

    # Step 2: Convert the 'Date' column to datetime format and sort by date
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
    data = data.sort_values(by='Date')

    # Step 3: Check if the dataset contains all required columns
    if not all(col in data.columns for col in feature_cols + [target_col]):
        raise ValueError("The dataset is missing required columns. Please check your file.")

    # Step 4: Select the specified feature and target columns, and drop any rows with missing values
    data = data[feature_cols + [target_col]].dropna()

    # Step 5: Normalize the data using MinMaxScaler
    # This scales all features to a range of 0 to 1, which helps LSTM models converge faster
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Step 6: Create overlapping input sequences and corresponding target values
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        # The input sequence consists of the previous 'sequence_length' rows
        X.append(scaled_data[i-sequence_length:i, :-1])
        # The target value is the corresponding value in the target column
        y.append(scaled_data[i, -1])

    # Convert the lists to NumPy arrays for easier manipulation
    X, y = np.array(X), np.array(y)

    # Step 7: Split the data into training and validation sets
    split_idx = int(len(X) * split_ratio)  # Index for splitting the data
    X_train, y_train = X[:split_idx], y[:split_idx]  # Training data
    X_val, y_val = X[split_idx:], y[split_idx:]      # Validation data

    # Convert the data to PyTorch tensors
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
        scaler  # Return the scaler for future use (e.g., scaling test data)
    )
