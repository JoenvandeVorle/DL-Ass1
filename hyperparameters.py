import torch
import torch.nn as nn


class Hyperparameters:
    window_size: int
    learning_rate: float
    optimizer: nn.Module
    loss_function: nn.Module
    epochs: int

    def __init__(self, **kwargs): # Allow dynamic initialization of any parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        def format_value(key, value):
            if key == "optimizer" and isinstance(value, torch.optim.Optimizer):
                return value.__class__.__name__  # Return only the class name of the optimizer
            return value

        return f"Hyperparameters({', '.join(f'{key}={format_value(key, value)}' for key, value in self.__dict__.items())})"

    def to_dict(self):
        return self.__dict__.copy()
