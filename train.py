import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hyperparameters import Hyperparameters
from log_level import LogLevel
from numpy import arange

PATIENCE = 10

def train(model: nn.Module, train_data: DataLoader, val_data: DataLoader, epochs: int, hp: Hyperparameters) -> dict:
    epochs_since_last_improvement = 0
    best_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        if LogLevel.LEVEL >= LogLevel.Level.INFO:
            print(f'\nEpoch {epoch + 1}/{epochs}')

        model.train()
        i = 0
        train_loss = 0
        for inputs, target in train_data:
            # Forward pass
            hp.optimizer.zero_grad()
            output = 0
            # for input_point in inputs[0]:
            output = model(inputs)
            loss = hp.loss_function(output, target)
            train_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            hp.optimizer.step()
            if LogLevel.LEVEL >= LogLevel.Level.VERBOSE:
                if i % 100 == 0:
                    print(f'Step [{i}], Loss: {loss.item():.4f}')
            i += 1

        # test on validation set and calculate losses
        avg_val_loss = test(val_data, model, hp)
        avg_train_loss = train_loss / len(train_data)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        print(f"Finished epoch {epoch + 1}: train loss {avg_train_loss:.4f}, val loss {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_since_last_improvement = 0
            if LogLevel.LEVEL > LogLevel.Level.INFO:
                print(f'Validation Loss improved to {best_loss:.4f}')
        else:
            epochs_since_last_improvement += 1
            if epochs_since_last_improvement > PATIENCE:
                if LogLevel.LEVEL >= LogLevel.Level.INFO:
                    print('Early stopping...')
                break

    train_results = {
        "epoch" : arange(0, len(train_losses)),
        "train_losses": train_losses,
        "val_losses": val_losses,
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
