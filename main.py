#!/usr/bin/env python3
"""
Module Docstring
"""

__author__ = "Joris Peeters, Sukru Saygili, Abdelmalek Matoug"
__version__ = "0.1.0"
__license__ = "GPLv3"

import argparse
import os
import torch
from model import LSTMModel
from data_utils import load_data
from train import perform_grid_search, predict_future, set_seed
from plot_utils import plot_predictions, plot_residuals
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error


def load_and_preprocess_data(training_file, feature_cols, target_col, sequence_length, split_ratio):
    """
    Load and preprocess the training data.
    
    This function reads the dataset, extracts the required features and target column,
    applies scaling, and splits the data into training and validation sets.
    """
    print("Loading and preprocessing training data...")
    return load_data(
        training_file, feature_cols=feature_cols, target_col=target_col,
        sequence_length=sequence_length, split_ratio=split_ratio
    )


def load_or_train_model(response, device, X_train, y_train, X_val, y_val, scaler, param_grid):
    """
    Load an existing model or train a new one based on the user response.
    
    If the user opts to train a new model, grid search is used to find the best parameters.
    If a saved model exists and is chosen, it will be loaded from disk.
    """
    model = None

    if response == "yes":
        print("Training a new model...")
        # Perform grid search to find the best hyperparameters and train the model
        best_model, best_params = perform_grid_search(
            param_grid, X_train, y_train, X_val, y_val, X_train.shape[2], scaler
        )
        print(f"Best hyperparameters: {best_params}")
        model = best_model
    else:
        print("Loading existing model...")
        # Load the saved model if it exists
        if os.path.exists('best_model.pth'):
            checkpoint = torch.load('best_model.pth', map_location=device)
            config = checkpoint['config']
            model = LSTMModel(
                input_size=config['input_size'],
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                output_size=config['output_size']
            ).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Error: No existing model found. Exiting...")
            return None

    return model


def prepare_test_input(X_train, model, test_data, scaler, feature_cols, sequence_length):
    """
    Prepare the test input for predictions.
    
    This function typically uses the last sequence of training data as input
    to seed the model's predictions.
    """
    print("Preparing test input...")
    if model is None:
        print("Error: Model is not initialized. Exiting.")
        return None

    # Default behavior: Use the last sequence of training data
    return X_train[-1, :, :].clone().detach()


def make_predictions(model, test_input, test_data, scaler, feature_cols):
    """
    Generate predictions and calculate the Mean Absolute Percentage Error (MAPE).
    
    The function takes a trained model and test input, generates predictions,
    and compares them to the actual values from the test dataset.
    """
    print("Generating predictions...")
    test_predictions = predict_future(
        model, test_input, len(test_data), scaler, feature_cols
    )

    test_actual = test_data['Last Close'].values
    mape = mean_absolute_percentage_error(test_actual, test_predictions)
    return test_predictions, test_actual, mape


def sp500_prediction(training_file, testing_file, param_grid):
    """
    Main SP500 prediction pipeline.
    
    This function coordinates the overall process: loading data, training or loading a model,
    preparing the test input, generating predictions, and calculating MAPE.
    """
    set_seed(42)  # Ensure reproducibility
    feature_cols = ['Open', 'High', 'Low', 'Last Close']
    target_col = 'Last Close'
    sequence_length = 20
    split_ratio = 0.987  # Use this ratio to ensure October 2024 is validation data

    # Load and preprocess the training data
    X_train, y_train, X_val, y_val, scaler = load_and_preprocess_data(
        training_file, feature_cols, target_col, sequence_length, split_ratio
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check if a saved model exists and ask the user whether to train or load it
    if os.path.exists('best_model.pth'):
        response = input(
            "A saved model exists. "
            "This model was made using October 2024"
            "as validation data for tuning the parameters.\n" 
            "If you use October 2024 as test data,"
            "the MAPE will be around 0.\n"
            "The split ratio(0.987) was calculated so October 2024 is used as validation data.\n"
            "So you should type yes and train a new model. " 
            "Or you can use different test data (for example November 2024).\n"
            "Do you want to train a new model? (yes/no):"
        ).strip().lower()
    else:
        response = "yes"

    # Train or load the model based on user input
    model = load_or_train_model(response, device, X_train, y_train, X_val, y_val, scaler, param_grid)

    # Load test data
    test_data = pd.read_csv(testing_file)
    test_dates = pd.to_datetime(test_data['Date'], format='%m/%d/%Y')

    # Prepare test input and generate predictions
    test_input = prepare_test_input(X_train, model, test_data, scaler, feature_cols, sequence_length)
    test_predictions, test_actual, mape = make_predictions(model, test_input, test_data, scaler, feature_cols)

    # Plot predictions
    plot_predictions(
        test_dates, test_actual, test_predictions, "Predictions", "predictions.png"
    )
    # Plot Residuals
    plot_residuals(
        test_dates, test_actual, test_predictions,
        "Residuals (Prediction Errors)",
        "residuals_plot.png"
    )
    return mape


def main(args):
    """
    Main function to execute the SP500 prediction.
    
    This function parses the arguments and runs the prediction pipeline.
    """
    training_file = args.training_file
    testing_file = args.testing_file

    # Define the hyperparameter grid for grid search
    param_grid = {
        'hidden_size': [32],
        'num_layers': [2],
        'learning_rate': [0.001],
        'batch_size': [16],
        'num_epochs': [100]
    }
    
    # Run the prediction pipeline and display the MAPE
    mape = sp500_prediction(training_file, testing_file, param_grid)
    print(f"MAPE: {mape:.4f}")


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("training_file", help="Training data file")
    parser.add_argument("testing_file", help="Testing data file")

    # Specify output of "--version"
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__))

    args = parser.parse_args()
    main(args)
