import torch
from torch.utils.data import Dataset, DataLoader, random_split
import scipy
import numpy as np
from hyperparameters import Hyperparameters as hp

from const import SCALING_FACTOR
WINDOW_SIZE = 5 # should probably use hyperparameters class

class CustomDataset(Dataset):
    def __init__(self, x, window_size):
        self.x = x # data points
        self.window_size = window_size

    def __getitem__(self, index):
        # Return list of n input data points and the subsequent ground truth point
        return torch.tensor(self.x[index : index + self.window_size]), torch.tensor(self.x[index + self.window_size + 1])
  
    def __len__(self):
        # Return the total number of samples
        return len(self.x)
    

def load_data() -> tuple[DataLoader, DataLoader]:
    mat_data = scipy.io.loadmat('Xtrain.mat')
    # Load and squeeze the actual data
    data = mat_data['Xtrain']
    laser_data = np.squeeze(data)
    # normalization here? 
    laser_data = laser_data/SCALING_FACTOR

    dataset = CustomDataset(laser_data, WINDOW_SIZE)

    # split training and validation 80:20
    training_data, validation_data = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(training_data, batch_size=1)
    validation_data = DataLoader(validation_data, batch_size=1)

    print(len(training_data))
    print(len(validation_data))

    train_features, train_label = next(iter(train_dataloader))
    print("features:" + str(train_features))
    print("labels:" + str(train_label))

    return train_dataloader, validation_data