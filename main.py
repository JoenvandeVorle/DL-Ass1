import sys
import os

import numpy as np
import scipy
import torch
import torch.nn as nn
import pandas as pd

from model import RNN_Model, FeedForwardModel
from dataloader import load_data
from train import train, predict, predict_test
from hyperparameters import Hyperparameters
from itertools import product
from log_level import LogLevel
from itertools import product

from const import SCALING_FACTOR
from visualize import visualize_training, visualize_predictions
from dataloader import CustomDataset

DATA_DIR = "data"
CHECKPOINT = "checkpoints/FeedForward_weights_WS32_NAdam_LR0.001_L1Loss_50.pth"
FINAL_WINDOW_SIZE = 32

#HYPERPARAMETERS = {
#    "window_sizes": [2, 5, 10, 15, 20, 30, 50],
#    "optimizers": [torch.optim.Adam, torch.optim.SGD, torch.optim.NAdam],
#    "initial_learning_rates": [0.001, 0.01, 0.1],
#    "loss_functions": [nn.MSELoss(), nn.L1Loss(), nn.SmoothL1Loss()],
#    "epochs": [10, 20, 50, 100],
#}

# Best
HYPERPARAMETERS = {
   "window_sizes": [8,16,32,64],
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

        # model = RNN_Model(1, 10, window_size, 3)
        model = FeedForwardModel(window_size, 16, 3)

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
        filename = f"{DATA_DIR}/{model.modelname}_train_WS{window_size}_{optimizer.__name__}_LR{learning_rate}_{loss_function.__class__.__name__}_{epochs}.csv"
        training.to_csv(filename, index=False)
        visualize_training(filename)

        # Save weights
        save_dir = "checkpoints"
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{model.modelname}_weights_WS{window_size}_{optimizer.__name__}_LR{learning_rate}_{loss_function.__class__.__name__}_{epochs}.pth")
        torch.save(model.state_dict(), save_path)


def do_predict():
    train_data, val_data = load_data(FINAL_WINDOW_SIZE, device)
    # model = RNN_Model(1, 10, 1, 10)
    model = FeedForwardModel(FINAL_WINDOW_SIZE, 16, 3)
    model.load_state_dict(torch.load(CHECKPOINT))
    model.to(device)

    outputs, targets, mse, mae = predict(val_data, train_data, model)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")

    os.makedirs(DATA_DIR, exist_ok=True)

    # scale the output back up
    outputs = [output * SCALING_FACTOR for output in outputs]
    targets = [target * SCALING_FACTOR for target in targets]

    visualize_predictions(outputs, targets)

    results_df = pd.DataFrame(outputs)
    results_df.to_csv(f"{DATA_DIR}/predictions.csv", index=False)
    errors_df = pd.DataFrame({"MSE": [mse], "MAE": [mae]})
    errors_df.to_csv(f"{DATA_DIR}/errors.csv", index=False)

def do_predict_test():
    train_data, val_data = load_data(FINAL_WINDOW_SIZE, device)
    # model = RNN_Model(1, 10, 1, 10)
    model = FeedForwardModel(FINAL_WINDOW_SIZE, 16, 3)
    model.load_state_dict(torch.load(CHECKPOINT))
    model.to(device)

    mat_data_test = scipy.io.loadmat('Xtest.mat')
    # Load and squeeze the actual data
    test_data_raw = mat_data_test['Xtest']
    test_data_raw = np.squeeze(test_data_raw)
    # normalization
    test_data = CustomDataset(test_data_raw/300, FINAL_WINDOW_SIZE,device)

    outputs, targets, mse, mae = predict_test(test_data, train_data, model, FINAL_WINDOW_SIZE, device)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")

    os.makedirs(DATA_DIR, exist_ok=True)

    # scale the output back up
    outputs = [output * SCALING_FACTOR for output in outputs]
    targets = [target * SCALING_FACTOR for target in targets]
    mat_data_train = scipy.io.loadmat('Xtrain.mat')
    # Load and squeeze the actual data
    train_data_raw = mat_data_train['Xtrain']
    train_data_raw = np.squeeze(train_data_raw)

    full_data_pred = np.squeeze(train_data_raw).tolist()
    full_data_pred += outputs

    full_data_gt = train_data_raw.tolist()
    full_data_gt += test_data_raw.tolist()

    visualize_predictions(full_data_pred, full_data_gt)

    results_df = pd.DataFrame(outputs)
    results_df.to_csv(f"{DATA_DIR}/predictions.csv", index=False)
    errors_df = pd.DataFrame({"MSE": [mse], "MAE": [mae]})
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

    # do_train()
    do_predict_test()
