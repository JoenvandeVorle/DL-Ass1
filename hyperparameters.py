import torch
import torch.nn as nn

class Hyperparameters:

    window_size: int
    optimizer: torch.optim.Optimizer
    loss_function: nn.Module
    # weight of previous inputs in the window

    def __init__(self, window_size: int, optimizer: torch.optim.Optimizer, loss_function: nn.Module):
        self.window_size = window_size
        self.optimizer = optimizer
        self.loss_function = loss_function