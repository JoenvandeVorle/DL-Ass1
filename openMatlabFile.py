import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the .mat file
mat_data = scipy.io.loadmat('Xtrain.mat')

# Check available variable names
print(mat_data.keys())


# Load and squeeze the actual data
data = mat_data['Xtrain']


# Reshape to 2D for scaler
data = data.reshape(-1, 1)

# Create and fit scaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Flatten back to 1D if needed
scaled_data = scaled_data.flatten()

print("Scaled Data:", scaled_data)

laser_data = np.squeeze(scaled_data)

# Plot the data as points
plt.figure(figsize=(10, 4))
plt.scatter(range(len(laser_data)), laser_data, s=10, c='blue', marker='o')  # 's' is size
plt.title('Laser Measurements (Scatter)')
plt.xlabel('Time Step')
plt.ylabel('Distance (or relevant unit)')
plt.grid(True)
plt.tight_layout()
plt.show()
