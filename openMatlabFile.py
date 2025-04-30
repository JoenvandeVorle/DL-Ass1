import scipy.io
import matplotlib.pyplot as plt
import numpy as np

# Load the .mat file
mat_data = scipy.io.loadmat('Xtrain.mat')

# Check available variable names
print(mat_data.keys())

# Load and squeeze the actual data
data = mat_data['Xtrain']
laser_data = np.squeeze(data)

# Plot the data as points
plt.figure(figsize=(10, 4))
plt.scatter(range(len(laser_data)), laser_data, s=10, c='blue', marker='o')  # 's' is size
plt.title('Laser Measurements (Scatter)')
plt.xlabel('Time Step')
plt.ylabel('Distance (or relevant unit)')
plt.grid(True)
plt.tight_layout()
plt.show()
