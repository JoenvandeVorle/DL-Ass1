import torch
import torch.nn as nn
import pandas as pd

from model import Model
from dataPreProcessing import load_data
from train import train
from hyperparameters import Hyperparameters
from itertools import product


HYPERPARAMETERS = {
    "window_sizes": [2, 5, 10, 15, 20, 30, 50],
    "optimizers": [torch.optim.Adam, torch.optim.SGD, torch.optim.NAdam],
    "initial_learning_rates": [0.001, 0.01, 0.1],
    "loss_functions": [nn.MSELoss(), nn.L1Loss(), nn.SmoothL1Loss()],
    "epochs": [100, 200, 300],
}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    activation_function = nn.ReLU()

    hyperparameter_combinations = product(
        HYPERPARAMETERS["window_sizes"],
        HYPERPARAMETERS["optimizers"],
        HYPERPARAMETERS["initial_learning_rates"],
        HYPERPARAMETERS["loss_functions"],
        HYPERPARAMETERS["epochs"],
    )

    for window_size, optimizer, learning_rate, loss_function, epochs in hyperparameter_combinations:
        train_data, val_data = load_data(window_size, device)

        model = Model(window_size, activation_function)
        model.to(device)

        opt = optimizer(model.parameters(), lr=learning_rate)
        hyperparameters = Hyperparameters(
            window_size=window_size,
            optimizer=opt,
            initial_learning_rate=learning_rate,
            loss_function=loss_function,
            epochs=epochs,
        )
        print(f"Training with params: {hyperparameters}")
        print(f"Params dict: {hyperparameters.to_dict()}")

        train_results = train(model, train_data, val_data, epochs, hyperparameters)
