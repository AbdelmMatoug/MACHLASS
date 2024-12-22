import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the data
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True)
    return data

# Preprocess the data: Extract features and target, with added lag features
def preprocess_data(data, feature_cols, target_col, lag_days=1):
    # Create lag features for the previous days
    for lag in range(1, lag_days + 1):
        for col in feature_cols:
            data[f'{col}_lag_{lag}'] = data[col].shift(lag)
    
    # Drop rows with NaN values (created by shifting)
    data = data.dropna()
    
    # Features and target
    X = data[[f'{col}_lag_{lag}' for col in feature_cols for lag in range(1, lag_days + 1)]].values
    y = data[target_col].values
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

# Train the SVR model
def train_svr_model(X_train, y_train, C=1.0, epsilon=0.1):
    svr = SVR(C=C, epsilon=epsilon)
    svr.fit(X_train, y_train)
    return svr

# Calculate MAPE
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Main function to run the model
def main(training_file, testing_file):
    # Load the training and testing data
    train_data = load_data(training_file)
    test_data = load_data(testing_file)

    # Define feature columns and target column
    feature_cols = ['Open', 'High', 'Low']
    target_col = 'Last Close'

    # Preprocess the training and testing data (using 1 lag feature)
    X_train, y_train, scaler = preprocess_data(train_data, feature_cols, target_col, lag_days=1)
    X_test, y_test, _ = preprocess_data(test_data, feature_cols, target_col, lag_days=1)

    # Train the SVR model
    svr_model = train_svr_model(X_train, y_train, C=1.0, epsilon=0.1)

    # Make predictions
    y_pred = svr_model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = calculate_mape(y_test, y_pred)

    # Print the evaluation results
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'R² Score: {r2}')
    print(f'MAPE: {mape}%')

    # Plot the predicted vs actual Last Close values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual Last Close', color='blue')
    plt.plot(y_pred, label='Predicted Last Close', color='red', linestyle='--')
    plt.xlabel('Test Samples')
    plt.ylabel('Last Close')
    plt.title('Actual vs Predicted Last Close')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_file', type=str, required=True, help="Path to the training data CSV file")
    parser.add_argument('--testing_file', type=str, required=True, help="Path to the testing data CSV file")
    args = parser.parse_args()

    # Run the main function
    main(args.training_file, args.testing_file)
