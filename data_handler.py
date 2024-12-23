import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

class DataHandler:
    
    def check_data_leakage(train_data, test_data, unique_col='Date'):
        train_values = set(train_data[unique_col])
        test_values = set(test_data[unique_col])
        overlap = train_values.intersection(test_values)
        if overlap:
            raise ValueError(f"Data leakage detected! Overlap in {unique_col}: {overlap}")
        else:
            print(f"No data leakage detected. Training and testing datasets are distinct.")

   
    def load_data(file_path, sequence_length=60, feature_cols=None, target_col=None, split_ratio=0.98):
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
        data = data.sort_values(by='Date')

        if not all(col in data.columns for col in feature_cols + [target_col]):
            raise ValueError("The dataset is missing required columns. Please check your file.")

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
            raise ValueError("The split ratio resulted in insufficient training or validation data.")

        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]

        return (
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
            scaler
        )
