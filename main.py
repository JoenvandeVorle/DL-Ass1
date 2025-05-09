import sys
import os

import torch
import torch.nn as nn
import pandas as pd

from model import RNN_Model, FeedForwardModel
from dataloader import load_data
from train import train, predict
from hyperparameters import Hyperparameters
from itertools import product
from log_level import LogLevel
from itertools import product

from const import SCALING_FACTOR
from visualize import visualize_training, visualize_predictions

DATA_DIR = "data"
CHECKPOINT = "checkpoints/RNN_weights_WS5_NAdam_LR0.001_L1Loss_30.pth"
FINAL_WINDOW_SIZE = 5
FINAL_MODEL = RNN_Model(1, 8, 2)
# model = FeedForwardModel(FINAL_WINDOW_SIZE, 10, 3)

HYPERPARAMETERS = {
    "window_sizes": [4],
    "optimizers": [torch.optim.NAdam],
    "initial_learning_rates": [0.001],
    "loss_functions": [nn.L1Loss()],
    "epochs": [30],
    "hidden_sizes": [8],
    "num_layers": [3],
}

# Best
HYPERPARAMETERS = {
    "window_sizes": [4, 8, 16, 32],
    "optimizers": [torch.optim.NAdam],
    "initial_learning_rates": [0.001],
    "loss_functions": [nn.L1Loss()],
    "epochs": [30],
    "hidden_sizes": [8, 16, 32, 64, 128, 255],
    "num_layers": [2, 3, 4, 5],
}

def do_train():
    hyperparameter_combinations = product(
        HYPERPARAMETERS["window_sizes"],
        HYPERPARAMETERS["optimizers"],
        HYPERPARAMETERS["initial_learning_rates"],
        HYPERPARAMETERS["loss_functions"],
        HYPERPARAMETERS["epochs"],
        HYPERPARAMETERS["hidden_sizes"],
        HYPERPARAMETERS["num_layers"],
    )

    for window_size, optimizer, learning_rate, loss_function, epochs, hidden_sz, num_ly in hyperparameter_combinations:
        train_data, val_data = load_data(window_size, device, randomsplit=True)

        model = RNN_Model(1, hidden_sz, num_ly)
        model.init_weights()
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
        model_name = f"{model.modelname}_train_WS{window_size}_HID{hidden_sz}_LYR{num_ly}"
        filename = f"{DATA_DIR}/{model_name}.csv"
        training.to_csv(filename, index=False)
        visualize_training(filename)

        # Save weights
        save_dir = "checkpoints"
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{model_name}.pth")
        torch.save(model.state_dict(), save_path)

        do_predict(model, window_size)


def do_predict(model: nn.Module = None, window_size: int = FINAL_WINDOW_SIZE):
    train_data, val_data = load_data(window_size, device, randomsplit=False)
    if model is None:
        model = RNN_Model(1, 8, 2)
        model.load_state_dict(torch.load(CHECKPOINT))

    model.to(device)

    outputs, targets, mse, mae = predict(val_data, model)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")

    os.makedirs(DATA_DIR, exist_ok=True)

    # scale the output back up
    outputs = [output * SCALING_FACTOR for output in outputs]
    targets = [target * SCALING_FACTOR for target in targets]

    model_name = f"{model.modelname}_predict_WS{window_size}_HID{model.hidden_size}_LYR{model.num_layers}"
    visualize_predictions(outputs, targets, model_name)

    results_df = pd.DataFrame(outputs)
    results_df.to_csv(f"{DATA_DIR}/{model.modelname}_predictions.csv", index=False)
    errors_df = pd.DataFrame({"MSE": [mse], "MAE": [mae]})
    errors_df.to_csv(f"{DATA_DIR}/{model.modelname}_errors.csv", index=False)


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
