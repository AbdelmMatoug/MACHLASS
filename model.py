import torch
import torch.nn as nn

# Define LSTM model
class LSTMModel(nn.Module):
    """
    LSTM-based neural network for time-series prediction.

    This model consists of:
    1. An LSTM layer to capture temporal dependencies in the input data.
    2. A fully connected (linear) layer to produce the final output.

    Args:
        input_size (int): Number of input features for each time step.
        hidden_size (int): Number of units in the LSTM's hidden layer.
        num_layers (int): Number of stacked LSTM layers.
        output_size (int): Number of output features (e.g., 1 for regression).

    """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()

        # Store configuration parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define an LSTM layer:
        # - input_size: Number of input features at each time step.
        # - hidden_size: Number of units in the hidden state.
        # - num_layers: Number of stacked LSTM layers.
        # - batch_first: Ensures input/output tensors are shaped as (batch, seq, feature).
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Define a fully connected (linear) layer:
        # This maps the LSTM's hidden state at the final time step to the output.
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the LSTM model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, output_size).
        """

        # Initialize the hidden state (h0) and cell state (c0) with zeros:
        # - num_layers: Number of LSTM layers.
        # - batch_size: Number of sequences in the batch.
        # - hidden_size: Size of each hidden state.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pass the input through the LSTM layer:
        # - out: Contains the output features (hidden states) for all time steps.
        # - _: Contains the last hidden state (h_n) and cell state (c_n), which are unused here.
        out, _ = self.lstm(x, (h0, c0))

        # Pass the final hidden state (last time step) through the fully connected layer:
        # - out[:, -1, :]: Select the output of the last time step for each sequence in the batch.
        return self.fc(out[:, -1, :])
