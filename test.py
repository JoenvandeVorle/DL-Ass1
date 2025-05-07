import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy
import matplotlib.pyplot as plt

from model import Model
from dataPreProcessing import load_data
from const import SCALING_FACTOR

window_size = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

mat_data = scipy.io.loadmat('Xtrain.mat')
# Load and squeeze the actual data
data = mat_data['Xtrain']
laser_data = np.squeeze(data)
# normalization
laser_data = laser_data/SCALING_FACTOR

model = Model(1, window_size)
model.to(device)

initial_input = laser_data[:window_size]
output = []
output.append(model(initial_input))
for i in range(1000):
    output = model(output[-1])

plt.figure(figsize=(10, 6))
plt.plot(y[0].numpy(), label='True')
plt.plot(X[0], label='Predicted')
plt.legend()
plt.show()