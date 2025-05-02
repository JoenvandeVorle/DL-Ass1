import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hyperparameters import Hyperparameters


def train(model: nn.Module, train_data: DataLoader, val_data: DataLoader, epochs: int, hp: Hyperparameters) -> dict:
    for epoch in range(epochs):
        model.train()
        i = 0
        for inputs, target in train_data:
            # Forward pass
            hp.optimizer.zero_grad()
            outputs = model(inputs)
            loss = hp.loss_function(outputs, target)

            # Backward pass and optimization
            loss.backward()
            hp.optimizer.step()
            if i % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i}], Loss: {loss.item():.4f}')
            i += 1

        # test on validation set
        avg_loss = test(val_data, model, hp)
        print(f'Validation Loss after epoch {epoch + 1}: {avg_loss:.4f}')

    train_results = {
        "avg_loss": avg_loss
        # add more...
    }
    return train_results


def test(test_set: DataLoader, model: nn.Module, hp: Hyperparameters) -> float:
    model.eval()
    with torch.no_grad():
        losses = []
        for inputs, target in test_set:
            output = model(inputs)
            loss = hp.loss_function(output, target)
            losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)
    return avg_loss
