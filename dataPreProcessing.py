import torch
from torch.utils.data import Dataset, DataLoader, random_split
import scipy
import numpy as np
from hyperparameters import Hyperparameters as hp

from const import SCALING_FACTOR
from log_level import LogLevel

class CustomDataset(Dataset):
    def __init__(self, x, window_size: int, device: torch.device):
        self.x = x # data points
        self.window_size = window_size
        self.device = device

    def __getitem__(self, index):
        # Return list of n input data points and the subsequent ground truth point
        return torch.tensor(self.x[index : index + self.window_size], dtype=torch.float32, device=self.device), \
            torch.tensor(self.x[index + self.window_size + 1], dtype=torch.float32, device=self.device)
  
    def __len__(self):
        # Return the total number of samples
        return len(self.x)

def load_data(window_size: int, device: torch.device) -> tuple[DataLoader, DataLoader]:
    mat_data = scipy.io.loadmat('Xtrain.mat')
    # Load and squeeze the actual data
    data = mat_data['Xtrain']
    laser_data = np.squeeze(data)
    # normalization
    laser_data = laser_data/SCALING_FACTOR

    dataset = CustomDataset(laser_data, window_size, device)

    # split training and validation 80:20
    # training_data, validation_data = random_split(dataset, [0.8, 0.2])
    train_last_index = int(len(dataset) * 0.8 - window_size - 1)
    training_data = []
    validation_data = []
    for i in range(len(dataset) - window_size - 1):
        if i < train_last_index:
            training_data.append(dataset[i])
        else:
            validation_data.append(dataset[i])
    train_dataloader = DataLoader(training_data, batch_size=1)
    validation_data = DataLoader(validation_data, batch_size=1)

    if LogLevel.LEVEL >= LogLevel.Level.VERBOSE:
        print(len(training_data))
        print(len(validation_data))

    train_features, train_label = next(iter(train_dataloader))
    if LogLevel.LEVEL >= LogLevel.Level.VERBOSE:
        print("train_features:" + str(train_features))
        print("train_label:" + str(train_label))

    return train_dataloader, validation_data
