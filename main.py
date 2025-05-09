import sys
import os

import torch
import torch.nn as nn
import pandas as pd

from model import Model
from dataloader import load_data
from train import train, predict
from hyperparameters import Hyperparameters
from itertools import product
from log_level import LogLevel
from itertools import product

DATA_DIR = "data"
CHECKPOINT = "checkpoints/RNN_weights_win8.pth"
FINAL_WINDOW_SIZE = 8

#HYPERPARAMETERS = {
#    "window_sizes": [2, 5, 10, 15, 20, 30, 50],
#    "optimizers": [torch.optim.Adam, torch.optim.SGD, torch.optim.NAdam],
#    "initial_learning_rates": [0.001, 0.01, 0.1],
#    "loss_functions": [nn.MSELoss(), nn.L1Loss(), nn.SmoothL1Loss()],
#    "epochs": [10, 20, 50, 100],
#}

# Best
HYPERPARAMETERS = {
   "window_sizes": [5],
   "optimizers": [torch.optim.NAdam],
   "initial_learning_rates": [0.001],
   "loss_functions": [nn.L1Loss()],
   "epochs": [50],
}

def do_train():
    hyperparameter_combinations = product(
        HYPERPARAMETERS["window_sizes"],
        HYPERPARAMETERS["optimizers"],
        HYPERPARAMETERS["initial_learning_rates"],
        HYPERPARAMETERS["loss_functions"],
        HYPERPARAMETERS["epochs"],
    )

    for window_size, optimizer, learning_rate, loss_function, epochs in hyperparameter_combinations:
        train_data, val_data = load_data(window_size, device)

        model = Model(1, 10, window_size, 10)
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

        train_results = train(model, train_data, val_data, epochs, hyperparameters)

        os.makedirs(DATA_DIR, exist_ok=True)
        # save train results except for outputs
        training = pd.DataFrame(train_results)
        training.to_csv(f"{DATA_DIR}/train_WS{window_size}_{optimizer.__name__}_LR{learning_rate}_{loss_function.__class__.__name__}_{epochs}.csv", index=False)

        # Save weights
        save_dir = "checkpoints"
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"RNN_weights_WS{window_size}_{optimizer.__name__}_LR{learning_rate}_{loss_function.__class__.__name__}_{epochs}.pth")
        torch.save(model.state_dict(), save_path)


def do_predict():
    train_data, val_data = load_data(FINAL_WINDOW_SIZE, device)
    model = Model(1, FINAL_WINDOW_SIZE)
    model.load_state_dict(torch.load(CHECKPOINT))
    model.to(device)

    outputs, mse, mae = predict(val_data, model)

    results_df = pd.DataFrame(outputs)
    results_df.to_csv(f"{DATA_DIR}/predictions.csv", index=False)
    errors_df = pd.DataFrame({"MSE": mse, "MAE": mae})
    errors_df.to_csv(f"{DATA_DIR}/errors.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        print("Usage: python main.py <log_level>")
        sys.exit(1)
    elif len(sys.argv) < 2:
        LogLevel.set_level(4)
    else:
        LogLevel.set_level(int(sys.argv[1]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    activation_function = nn.ReLU()

    do_train()
    # do_predict()
