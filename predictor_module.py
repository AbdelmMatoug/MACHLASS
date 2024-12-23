import numpy as np
import torch
import matplotlib.pyplot as plt

class Predictor:
    
    def predict_future(model, input_data, prediction_steps, scaler, feature_cols):
        model.eval()
        predictions = []
        current_input = input_data.clone()

        with torch.no_grad():
            for step in range(prediction_steps):
                prediction = model(current_input.unsqueeze(0)).squeeze(-1)
                predictions.append(prediction.item())
                next_input = torch.zeros_like(current_input)
                next_input[:-1, :] = current_input[1:, :]
                next_input[-1, :] = prediction
                current_input = next_input

        predictions = np.array(predictions).reshape(-1, 1)
        inv_predictions = np.zeros((len(predictions), len(feature_cols) + 1))
        inv_predictions[:, -1] = predictions[:, 0]
        inv_predictions = scaler.inverse_transform(inv_predictions)

        return inv_predictions[:, -1]

    
    def plot_predictions(dates, actual, predicted, title, filename):
        plt.figure(figsize=(12, 6))
        plt.plot(dates, actual, label='Actual', marker='o', color='blue')
        plt.plot(dates, predicted, label='Predicted', marker='x', color='orange')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.show()
        plt.close()
