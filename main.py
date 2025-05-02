import torch
import torch.nn as nn
import pandas as pd

from const import EPOCHS
from model import Model
from dataPreProcessing import load_data
from train import train
from hyperparameters import Hyperparameters


HYPERPARAMETERS = {
    "window_sizes": [2, 5, 10, 15, 20, 30, 50],
    "optimizers": [torch.optim.Adam, torch.optim.SGD, torch.optim.NAdam],
    "loss_functions": [nn.MSELoss(), nn.L1Loss(), nn.SmoothL1Loss()],
}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    activation_function = nn.ReLU()

    for window_size in HYPERPARAMETERS["window_sizes"]:
        train_data, val_data = load_data(window_size, device)
        for optimizer in HYPERPARAMETERS["optimizers"]:
            for loss_function in HYPERPARAMETERS["loss_functions"]:
                hyperparameters = Hyperparameters(window_size, optimizer(model.parameters(), lr=0.001), loss_function)
                print(f"Training with params: {hyperparameters}")

                model = Model(window_size, activation_function)
                model.to(device)
                #model.display()

                train_results = train(model, train_data, val_data, EPOCHS, hyperparameters)
