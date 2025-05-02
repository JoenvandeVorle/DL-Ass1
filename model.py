import torch
import torch.nn as nn
from torchinfo import summary

class Model(nn.Module):

    window_size: int
    activation_function: nn.Module

    def __init__(self, window_size: int, activation_function: nn.Module):
        super(Model, self).__init__()
        self.window_size = window_size
        self.activation_function = activation_function
        
        self.rnn = nn.RNN(window_size, 32, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 32)
        # self.fc1 = nn.Linear(window_size, 32)
        # self.fc2 = nn.Linear(32, 64)
        # self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 32).to(x.device)
        x, _ = self.rnn(x, h0)

        x = x[:, -1, :]

        x = self.fc1(x)
        x = self.activation_function(x)
        x = self.fc2(x)
        x = self.activation_function(x)
        # x = self.fc3(x)
        # x = self.activation_function(x)

        x = self.output(x)
        x = torch.sigmoid(x)

        return x
    
    def display(self):
        summary(self, self.window_size)