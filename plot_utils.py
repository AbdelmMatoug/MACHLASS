import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(dates, actual, predicted, title, filename):
    """
    Plot actual vs. predicted values over time.
    
    Args:
        dates (list): List of dates for the x-axis.
        actual (list): Actual target values.
        predicted (list): Predicted values.
        title (str): Title of the plot.
        filename (str): Filename to save the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual', marker='o', color='blue')
    plt.plot(dates, predicted, label='Predicted', marker='x', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


def plot_training_validation_loss(train_losses, val_losses, title, filename):
    """
    Plot training and validation loss over epochs.
    
    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        title (str): Title of the plot.
        filename (str): Filename to save the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


def plot_residuals(dates, actual, predicted, title, filename):
    """
    Plot residuals (prediction errors) over time.
    
    Args:
        dates (list): List of dates for the x-axis.
        actual (list): Actual target values.
        predicted (list): Predicted values.
        title (str): Title of the plot.
        filename (str): Filename to save the plot.
    """
    residuals = np.array(actual) - np.array(predicted)
    plt.figure(figsize=(12, 6))
    plt.plot(dates, residuals, marker='o', color='red', label='Residuals')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel('Date')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()
