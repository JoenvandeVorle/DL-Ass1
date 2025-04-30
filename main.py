import torch
import torch.nn as nn

from const import EPOCHS
from model import Model
from train import load_data, train
from hyperparameters import Hyperparameters


HYPERPARAMETERS = {
    "window_sizes": [10],
    "optimizers": [torch.optim.Adam],
    "loss_functions": [nn.MSELoss()],
}

if __name__ == "__main__":
    activation_function = nn.ReLU()

    data = load_data()
    for window_size in HYPERPARAMETERS["window_sizes"]:
        for optimizer in HYPERPARAMETERS["optimizers"]:
            for loss_function in HYPERPARAMETERS["loss_functions"]:
                print(f"Training with window size: {window_size}, optimizer: {optimizer}, loss function: {loss_function}")
                model = Model(window_size, activation_function)
                hyperparameters = Hyperparameters(window_size, optimizer(model.parameters(), lr=0.001), loss_function)
                train(model, data, EPOCHS, hyperparameters)
