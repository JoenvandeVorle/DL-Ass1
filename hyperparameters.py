import torch
import torch.nn as nn


class Hyperparameters:
    def __init__(self, **kwargs): # Allow dynamic initialization of any parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        return f"Hyperparameters({', '.join(f'{key}={value}' for key, value in self.__dict__.items())})"

    def to_dict(self):
        return self.__dict__.copy()
