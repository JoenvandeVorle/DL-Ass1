import torch
import torch.nn as nn
from torchinfo import summary

class Model(nn.Module):

    window_size: int
    activation_function: nn.Module

    def __init__(self, input_size, window_size: int):
        super(Model, self).__init__()
        self.window_size = window_size

        self.rnn = nn.RNN(1, 1, num_layers=10, dropout=0.2, batch_first=True)
        # self.fc1 = nn.Linear(window_size, 32)
        # self.fc2 = nn.Linear(32, 64)
        # self.fc3 = nn.Linear(64, 32)

    def forward(self, x):
        h0 = torch.zeros((10, 1), device=x.device)
        x = x.T
        x, _ = self.rnn(x, h0.squeeze(0))

        # print (f"RNN output shape: {x.shape} -- h0 shape: {h0.shape}")

        # x = self.output(x)
        return x

    def display(self):
        summary(self, self.window_size)
