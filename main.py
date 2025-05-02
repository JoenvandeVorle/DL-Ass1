import torch
import torch.nn as nn

from const import EPOCHS
from model import Model
from dataPreProcessing import load_data
from train import train
from hyperparameters import Hyperparameters


HYPERPARAMETERS = {
    "window_sizes": [10],
    "optimizers": [torch.optim.Adam],
    "loss_functions": [nn.MSELoss()],
}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    activation_function = nn.ReLU()

    train_data, val_data = load_data()
    for window_size in HYPERPARAMETERS["window_sizes"]:
        for optimizer in HYPERPARAMETERS["optimizers"]:
            for loss_function in HYPERPARAMETERS["loss_functions"]:
                print(f"Training with window size: {window_size}, optimizer: {optimizer}, loss function: {loss_function}")
                model = Model(window_size, activation_function)
                hyperparameters = Hyperparameters(window_size, optimizer(model.parameters(), lr=0.001), loss_function)
                train(model, train_data, val_data, EPOCHS, hyperparameters, device)
