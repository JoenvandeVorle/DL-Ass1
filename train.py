import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hyperparameters import Hyperparameters


def train(model: nn.Module, train_data: DataLoader, val_data: DataLoader, epochs: int, hp: Hyperparameters, device: torch.device) -> None:
    model.train()
    
    for epoch in range(epochs):
        i = 0
        for x, y in train_data:
            inputs, target = x.to(device), y.to(device)

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
        avg_loss = test(val_data, model, hp, device)
        print(f'Validation Loss after epoch {epoch + 1}: {avg_loss:.4f}')
        

def test(test_set: DataLoader, model: nn.Module, hp: Hyperparameters, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        losses = []
        for x, y in test_set:
            inputs, target = x.to(device), y.to(device)
            output = model(inputs)
            loss = hp.loss_function(output, target)
            losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)
    return avg_loss
