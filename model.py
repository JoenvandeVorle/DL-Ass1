import torch
import torch.nn as nn
from torchinfo import summary

class Model(nn.Module):

    input_size: int
    hidden_size: int
    window_size: int
    num_layers: int
    activation_function: nn.Module

    def __init__(self, input_size: int, hidden_size: int, window_size: int, num_layers: int):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        h0 = torch.zeros((self.num_layers, self.hidden_size), device=x.device)
        x = x.T
        x, _ = self.rnn(x, h0.squeeze(0))

        # print (f"RNN output shape: {x.shape} -- h0 shape: {h0.shape}")

        x = self.fc(x[-1])
        return x

    def display(self):
        summary(self, self.window_size)
