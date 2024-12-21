import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima

# Function to calculate Mean Absolute Percentage Error (MAPE)
def calculate_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100

# Load training and testing data
def load_data(train_file, test_file):
    # Read CSV files into pandas DataFrames
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Ensure proper datetime format for date column
    train_data['Date'] = pd.to_datetime(train_data['Date'])
    test_data['Date'] = pd.to_datetime(test_data['Date'])

    return train_data, test_data

# Preprocess data for time series modeling
def preprocess_data(train_data, test_data):
    # Set the 'Date' column as the index for both train and test data
    train_data = train_data.set_index('Date')
    test_data = test_data.set_index('Date')

    # Ensure that the Date index has a frequency (daily frequency assumed here)
    train_data = train_data.asfreq('D')
    test_data = test_data.asfreq('D')

    # Only use 'Last Close' for prediction
    y_train = train_data['Last Close']
    y_test = test_data['Last Close']
    
    return y_train, y_test

# Check if data is stationary using Augmented Dickey-Fuller (ADF) test
def check_stationarity(series):
    # Check for missing or infinite values
    if series.isnull().any():
        print("Data contains missing values. Filling missing values.")
        series = series.fillna(method='ffill')  # Forward fill to handle missing values
    
    if np.isinf(series).any():
        print("Data contains infinite values. Replacing infinite values.")
        series = series.replace([np.inf, -np.inf], np.nan)  # Replace infinite values with NaN
        series = series.fillna(method='ffill')  # Fill NaNs after replacing infinities
    
    # Perform ADF test to check stationarity
    result = adfuller(series)
    p_value = result[1]
    if p_value < 0.05:
        print("Data is stationary.")
    else:
        print("Data is non-stationary, differencing required.")
    return p_value < 0.05

# Train the ARIMA model (or SARIMA if seasonality is needed)
def train_arima_model(y_train, p=5, d=1, q=0, seasonal=False, m=12):
    # If data is non-stationary, apply differencing
    if not check_stationarity(y_train):
        y_train = y_train.diff().dropna()  # First difference
    
    if seasonal:
        # SARIMA model for seasonality
        model = ARIMA(y_train, order=(p, d, q), seasonal_order=(1, 1, 1, m))
    else:
        # Regular ARIMA model
        model = ARIMA(y_train, order=(p, d, q))
    
    model_fit = model.fit()
    return model_fit

# Auto ARIMA for parameter selection (optional)
def auto_arima_model(y_train):
    model = auto_arima(y_train, seasonal=True, m=12, trace=True, error_action='ignore', suppress_warnings=True)
    return model

# Make predictions and evaluate the model
def evaluate_model(model_fit, y_test):
    # Forecasting the values on the test set
    forecast = model_fit.forecast(steps=len(y_test))
    
    # Calculate MAPE
    mape = calculate_mape(y_test, forecast)
    print(f'MAPE on Test Data: {mape:.2f}%')
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test, label='Actual Last Close', color='blue')
    plt.plot(y_test.index, forecast, label='Predicted Last Close', color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted Last Close Price (ARIMA/SARIMA Model)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

    return forecast

# Main function to run the full process
def main(train_file, test_file, seasonal=False, m=12):
    # Load data
    train_data, test_data = load_data(train_file, test_file)

    # Preprocess data
    y_train, y_test = preprocess_data(train_data, test_data)

    # Train the ARIMA or SARIMA model
    model_fit = train_arima_model(y_train, p=5, d=1, q=0, seasonal=seasonal, m=m)

    # Evaluate the model
    forecast = evaluate_model(model_fit, y_test)

if __name__ == '__main__':
    # Replace with your actual file paths
    train_file = 'train.csv'  # Path to the training data file
    test_file = 'test.csv'    # Path to the testing data file
    main(train_file, test_file, seasonal=True)  # Set seasonal=True for SARIMA
