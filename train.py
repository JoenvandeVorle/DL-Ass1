import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hyperparameters import Hyperparameters
from log_level import LogLevel


def train(model: nn.Module, train_data: DataLoader, val_data: DataLoader, epochs: int, hp: Hyperparameters) -> None:
    epochs_since_last_improvement = 0
    best_loss = float('inf')
    for epoch in range(epochs):
        if LogLevel.LEVEL >= LogLevel.Level.INFO:
            print(f'Epoch {epoch + 1}/{epochs}')

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
            if LogLevel.LEVEL >= LogLevel.Level.VERBOSE:
                if i % 100 == 0:
                    print(f'Step [{i}], Loss: {loss.item():.4f}')
            i += 1

        # test on validation set
        avg_loss = test(val_data, model, hp)
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_since_last_improvement = 0
            if LogLevel.LEVEL >= LogLevel.Level.INFO:
                print(f'Validation Loss improved to {best_loss:.4f}')
        else:
            epochs_since_last_improvement += 1
            if epochs_since_last_improvement > 10:
                if LogLevel.LEVEL >= LogLevel.Level.INFO:
                    print('Early stopping...')
                break


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
