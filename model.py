import torch
import torch.nn as nn
from torchinfo import summary

class Model(nn.Module):

    window_size: int
    activation_function: nn.Module

    def __init__(self, input_size, window_size: int):
        super(Model, self).__init__()
        self.window_size = window_size
        
        self.rnn = nn.RNN(input_size, window_size, batch_first=True)
        # self.fc1 = nn.Linear(window_size, 32)
        # self.fc2 = nn.Linear(32, 64)
        # self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(window_size, 1)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.window_size, device=x.device)
        x, _ = self.rnn(x.unsqueeze(0), h0.squeeze(0))

        x = self.output(x)
        return x
    
    def display(self):
        summary(self, self.window_size)