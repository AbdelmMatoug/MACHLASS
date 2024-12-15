import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
import pmdarima as pm

def preprocess_data(file_path):
    """
    Preprocess the input data.
    """
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    features = ['Open', 'High', 'Low']
    target = 'Last Close'
    return data, features, target

def calculate_mape(training_file, testing_file):
    """
    Train models using the training data and calculate MAPE on the testing data.
    """
    # Load and preprocess the data
    training_data, features, target = preprocess_data(training_file)
    testing_data, _, _ = preprocess_data(testing_file)

    # Extract features and target
    X_train = training_data[features]
    y_train = training_data[target]
    X_test = testing_data[features]
    y_test = testing_data[target]

    # **Linear Regression Model**
    # Train the Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Predict using the Linear Regression model
    y_pred_lr = lr_model.predict(X_test)

    # Calculate MAPE for Linear Regression
    mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr)
    print(f"Linear Regression MAPE: {mape_lr * 100:.2f}%")

    # **ARIMA Model (Auto ARIMA)**
    # Automatically determine the best ARIMA parameters
    arima_model = pm.auto_arima(training_data[target], seasonal=True, stepwise=False, trace=True)
    print(f"Best ARIMA parameters: {arima_model.order}")

    # Predict using the ARIMA model
    y_pred_arima = arima_model.predict(n_periods=len(testing_data))

    # Calculate MAPE for ARIMA
    mape_arima = mean_absolute_percentage_error(y_test, y_pred_arima)
    print(f"ARIMA MAPE: {mape_arima * 100:.2f}%")

    # **Plotting: Actual vs Predicted Values for Linear Regression and ARIMA**
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label="Actual Values", color="blue")
    plt.plot(y_test.index, y_pred_lr, label="Linear Regression Predictions", color="green", linestyle='--')
    plt.plot(y_test.index, y_pred_arima, label="ARIMA Predictions", color="red", linestyle='--')
    plt.xlabel("Date")
    plt.ylabel("Last Close Price")
    plt.title("Actual vs Predicted Values for Linear Regression and ARIMA")
    plt.legend()
    plt.show()

    return mape_lr, mape_arima

def main():
    # Define your file paths here
    training_file = 'train.csv'
    testing_file = 'test.csv'
    
    # Calculate MAPE for both models
    mape_lr, mape_arima = calculate_mape(training_file, testing_file)

    print(f"Linear Regression MAPE: {mape_lr * 100:.2f}%")
    print(f"ARIMA MAPE: {mape_arima * 100:.2f}%")

if __name__ == "__main__":
    main()
