import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Function to calculate Mean Absolute Percentage Error (MAPE)
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Load training and testing data
def load_data(train_file, test_file):
    # Read CSV files into pandas DataFrames
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Ensure proper datetime format for date column
    train_data['Date'] = pd.to_datetime(train_data['Date'])
    test_data['Date'] = pd.to_datetime(test_data['Date'])

    return train_data, test_data

# Feature engineering and preprocessing
def preprocess_data(train_data, test_data):
    # We will use 'Open', 'High', and 'Low' prices as features for prediction
    features = ['Open', 'High', 'Low']
    target = 'Last Close'  # The target column for prediction is 'Last Close'

    # Extract features and target from the training and testing data
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]

    # Scaling the data (Standardization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test

# Train the model using Linear Regression
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Make predictions and evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    # Calculate MAPE
    mape = calculate_mape(y_test, y_pred)
    print(f'MAPE on Test Data: {mape:.2f}%')
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test, label='Actual Last Close', color='blue')
    plt.plot(y_test.index, y_pred, label='Predicted Last Close', color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted Last Close Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

    return y_pred

# Main function to run the full process
def main(train_file, test_file):
    # Load data
    train_data, test_data = load_data(train_file, test_file)

    # Preprocess data
    X_train, y_train, X_test, y_test = preprocess_data(train_data, test_data)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    y_pred = evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    # Replace with your actual file paths
    train_file = 'train.csv'  # Path to the training data file
    test_file = 'test.csv'    # Path to the testing data file
    main(train_file, test_file)
