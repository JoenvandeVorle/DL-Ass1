import torch
import torch.nn as nn
from torchinfo import summary

class RNN_Model(nn.Module):
    """
    A simple RNN model for time series prediction.
    """
    modelname = "RNN"
    input_size: int
    hidden_size: int
    num_layers: int
    activation_function: nn.Module

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(RNN_Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        # ensure output is 0-1
        # self.output = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros((self.num_layers, self.hidden_size), device=x.device)
        x = x.T
        x, _ = self.rnn(x, h0.squeeze(0))

        # print (f"RNN output shape: {x.shape} -- h0 shape: {h0.shape}")

        x = self.fc(x[-1])
        # x = self.output(x)
        return x

    def display(self):
        summary(self, self.window_size)

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)


class FeedForwardModel(nn.Module):
    """
    A simple feedforward model for time series prediction.

    It should take in a series of time steps and output a prediction for the next time step.
    """
    modelname = "FeedForward"
    input_size: int
    hidden_size: int
    num_layers: int

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(FeedForwardModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, input_size))
        self.model = nn.Sequential(*layers)

        self.output = nn.Linear(input_size, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        x = self.output(x)
        return x

    def display(self):
        summary(self, self.input_size)

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
