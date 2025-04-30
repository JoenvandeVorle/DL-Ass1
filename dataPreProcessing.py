import torch
from torch.utils.data import Dataset, random_split
import scipy
import numpy as np

# class CustomDataset(Dataset):
#     def __init__(self, x, y):
#         self.x = x # data point (k previous data points)
#         self.y = y # label (k+1 point)

#     def __getitem__(self, index):
#         # Return a dictionary with 'features' and 'label' as keys
#         return torch.tensor(self.x[index], dtype=torch.float32)
    
#     def __len__(self):
#         # Return the total number of samples
#         return len(self.x)
    
mat_data = scipy.io.loadmat('Xtrain.mat')
# Load and squeeze the actual data
data = mat_data['Xtrain']
laser_data = np.squeeze(data)

# split training and validation 80:20
trainingData = data[:799]
validationData = data[800:999]

# normalization here? 