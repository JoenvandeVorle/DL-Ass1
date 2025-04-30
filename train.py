import torch
import torch.nn as nn
import scipy.io

from const import SCALING_FACTOR
from hyperparameters import Hyperparameters


def load_data() -> torch.Tensor:
    # Load the .mat file
    mat_data = scipy.io.loadmat('Xtrain.mat')

    # Load the actual data
    data = mat_data['Xtrain']

    scaled_data = data/SCALING_FACTOR

    return scaled_data #, validation_data


def train(model: nn.Module, data: torch.Tensor, epochs: int, hp: Hyperparameters) -> None:
    model.train()
    
    for epoch in range(epochs):
        for i in range(data.shape[0] - hp.window_size):
            inputs = data[i:i + hp.window_size]
            target = data[i + hp.window_size]

            inputs = torch.tensor(inputs, dtype=torch.float32)
            target = torch.tensor(target, dtype=torch.float32)

            # Forward pass
            hp.optimizer.zero_grad()
            outputs = model(inputs)
            loss = hp.loss_function(outputs, target)

            # Backward pass and optimization
            loss.backward()
            hp.optimizer.step()
            if i % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i}], Loss: {loss.item():.4f}')

        # test on validation set
        avg_loss = test(data, model, hp)
        print(f'Validation Loss after epoch {epoch + 1}: {avg_loss:.4f}')
        

def test(test_set: torch.Tensor, model: nn.Module, hp: Hyperparameters) -> float:
    model.eval()
    with torch.no_grad():
        losses = []
        for i in range(test_set.shape[0] - hp.window_size):
            inputs = test_set[i:i + hp.window_size]
            inputs = torch.tensor(inputs, dtype=torch.float32)
            output = model(inputs)
            loss = hp.loss_function(output, test_set[i + hp.window_size])
            losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)
    return avg_loss
