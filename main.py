import sys

import torch
import torch.nn as nn

from model import Model
from dataPreProcessing import load_data
from train import train
from hyperparameters import Hyperparameters
from log_level import LogLevel


HYPERPARAMETERS = {
    "window_sizes": [2, 5, 10, 15, 20, 30, 50],
    "optimizers": [torch.optim.Adam, torch.optim.SGD, torch.optim.NAdam],
    "initial_learning_rates": [0.001, 0.01, 0.1],
    "loss_functions": [nn.MSELoss(), nn.L1Loss(), nn.SmoothL1Loss()],
    "epochs": [50, 100, 200, 300],
}

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <log_level>")
        sys.exit(1)
    LogLevel.set_level(int(sys.argv[1]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    activation_function = nn.ReLU()

    for window_size in HYPERPARAMETERS["window_sizes"]:
        train_data, val_data = load_data(window_size, device)
        for optimizer in HYPERPARAMETERS["optimizers"]:
            for learning_rate in HYPERPARAMETERS["initial_learning_rates"]:
                for loss_function in HYPERPARAMETERS["loss_functions"]:
                    for epochs in HYPERPARAMETERS["epochs"]:
                        print(f"Training with window size: {window_size}, optimizer: {optimizer}, learning_rate: {learning_rate}, loss function: {loss_function}")
                        model = Model(window_size, activation_function)
                        model.to(device)
                        #model.display()
                        opt = optimizer(model.parameters(), lr=learning_rate)
                        hyperparameters = Hyperparameters(window_size, opt, loss_function)
                        train(model, train_data, val_data, epochs, hyperparameters)
