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

    def __str__(self):
        return f"Hyperparameters(window_size={self.window_size}, optimizer={self.optimizer.__class__}, loss_function={self.loss_function})"

    def to_dict(self):
        return {
            "window_size": self.window_size,
            "optimizer": str(self.optimizer),
            "loss_function": str(self.loss_function)
        }
