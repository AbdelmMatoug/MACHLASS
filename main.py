import sys
import os
import argparse
import torch
import pandas as pd
from LSTMModel import LSTMModel
from data_handler import DataHandler
from trainer import Trainer
from predictor_module import Predictor
from sklearn.metrics import mean_absolute_percentage_error
from itertools import product


def main(args):
    feature_cols = ['Open', 'High', 'Low', 'Last Close']
    target_col = 'Last Close'

    # Load training data
    print("Loading and preprocessing training data...")
    X_train, y_train, X_val, y_val, scaler = DataHandler.load_data(
        args.training_file, feature_cols=feature_cols, target_col=target_col
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check if a saved model exists
    if os.path.exists('best_model.pth'):
        response = input(
            "A saved model exists. "
            "This model was made using October 2024"
            "as validation data for tuning the parameters.\n" 
            "If you use October 2024 as test data,"
            "the MAPE will be around 0.\n"
            "The split ratio(0.987) was calculated so october 2024 is used as validation data.\n"
            "So you should type yes and train a new model. " 
            "Or you can use different test data (for example November 2024). "
            "Do you want to train a new model? (yes/no):"
            
        ).strip().lower()
    else:
        response = "yes"

    # Train a new model or load the existing one
    if response == "yes":
        param_grid = {
            'hidden_size': [32, 64],
            'num_layers': [1, 2, 3],
            'learning_rate': [0.01, 0.001],
            'batch_size': [16, 32],
            'num_epochs': [50, 100]
        }

        print("Starting grid search for optimal parameters...")
        total_combinations = len(list(product(*param_grid.values())))
        combination_count = 0

        # Loop through all parameter combinations
        for params in product(*param_grid.values()):
            param_dict = dict(zip(param_grid.keys(), params))

            combination_count += 1
            # No combination print here; it will happen inside train_model
            print(f"Testing parameters: {param_dict}")

            # Instantiate model with current parameters
            model = LSTMModel(
                input_size=X_train.shape[2],
                hidden_size=param_dict['hidden_size'],
                num_layers=param_dict['num_layers'],
                output_size=1
            ).to(device)

            # Train the model with current parameters
            Trainer.train_model(
                model, X_train, y_train, X_val, y_val,
                num_epochs=param_dict['num_epochs'],
                learning_rate=param_dict['learning_rate'],
                batch_size=param_dict['batch_size'],
                combination_idx=combination_count,
                total_combinations=total_combinations
            )
    else:
        print("Loading existing model...")
        if os.path.exists('best_model.pth'):
            checkpoint = torch.load('best_model.pth', map_location=device, weights_only=True)
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
            return

    # Ensure the model variable is properly defined before prediction
    if 'model' not in locals():
        print("Error: Model not defined. Please train or load a model.")
        return

    # Load and process test data
    print("Loading and processing test data...")
    test_data = pd.read_csv(args.testing_file)
    test_dates = pd.to_datetime(test_data['Date'], format='%m/%d/%Y')
    test_actual = test_data[target_col].values

    # Generate predictions
    test_input = X_train[-1, :, :].clone().detach()
    test_predictions = Predictor.predict_future(
        model, test_input, len(test_data), scaler, feature_cols
    )

    # Plot predictions
    Predictor.plot_predictions(
        test_dates, test_actual, test_predictions,
        title="Predictions", filename="predictions.png"
    )

    # Calculate and print MAPE
    mape = mean_absolute_percentage_error(test_actual, test_predictions)
    print(f"Final Test MAPE: {mape * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("training_file", help="Path to the training data file")
    parser.add_argument("testing_file", help="Path to the testing data file")
    args = parser.parse_args()
    main(args)
