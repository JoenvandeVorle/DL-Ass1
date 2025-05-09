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
    mse_losses = []
    mae_losses = []

    for epoch in range(epochs):
        if LogLevel.LEVEL >= LogLevel.Level.INFO:
            print(f'\nEpoch {epoch + 1}/{epochs}')

        model.train()
        i = 0
        train_loss = 0
        for input_points, target in train_data:
            # Forward pass
            hp.optimizer.zero_grad()
            outputs = model(input_points) # model returns shape (window_size, batch, 1)
            # targets contains labels for all input points + the one at the end of the sequence that isn't used as input (y)
            targets = input_points[0, 1:]
            targets = torch.cat((targets, target))
            loss = hp.loss_function(outputs, targets)
            train_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            hp.optimizer.step()
            if LogLevel.LEVEL >= LogLevel.Level.VERBOSE:
                if i % 100 == 0:
                    print(f'Step [{i}], Loss: {loss.item():.4f}')
            i += 1

        # test on validation set and calculate losses
        avg_val_loss, avg_mse_loss, avg_mae_loss = test(val_data, model, hp)
        avg_train_loss = train_loss / len(train_data)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        mse_losses.append(avg_mse_loss)
        mae_losses.append(avg_mae_loss)
        print(f"Finished epoch {epoch + 1}: train loss {avg_train_loss:.4f}, val loss {avg_val_loss:.4f}")

        if avg_val_loss < (best_loss - 0.001):
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
        "mse_losses": mse_losses,
        "mae_losses": mae_losses,
    }
    return train_results


def test(test_set: DataLoader, model: nn.Module, hp: Hyperparameters) -> tuple[float, float, float]:
    model.eval()
    with torch.no_grad():
        losses = []
        mse_losses = []
        mae_losses = []
        for inputs, target in test_set:
            output = model(inputs)
            loss = hp.loss_function(output, target)
            mse_loss = nn.MSELoss()(output, target)
            mae_loss = nn.L1Loss()(output, target)
            losses.append(loss.item())
            mse_losses.append(mse_loss.item())
            mae_losses.append(mae_loss.item())
        avg_loss = sum(losses) / len(losses)
        avg_mse_loss = sum(mse_losses) / len(mse_losses)
        avg_mae_loss = sum(mae_losses) / len(mae_losses)
    return avg_loss, avg_mse_loss, avg_mae_loss


def predict(test_set: DataLoader, model: nn.Module) -> tuple[list[float], list[float], float, float]:
    model.eval()
    mae_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    mae_avg = 0
    mse_avg = 0
    with torch.no_grad():
        predictions = []
        targets = []
        i = 0
        output = 0
        outputs = None
        for data, target in test_set:
            if i == 0:
                input = data
                outputs = input
            else:
                # Pop and push
                outputs = outputs[:, 1:]  # Assuming outputs is a 2D tensor
                output_tensor = output.unsqueeze(1)  # Ensure the new output has the correct shape
                outputs = torch.cat((outputs, output_tensor), dim=1)
                input = torch.tensor(outputs)
            print (f"Input: {input}")
            output = model(input)
            print (f"Output: {output}")
            predictions.append(output.item())
            targets.append(target.item())
            mae_avg += mae_loss(output, data).item()
            mse_avg += mse_loss(output, data).item()
            i+= 1
        mae_avg /= len(test_set)
        mse_avg /= len(test_set)
        if LogLevel.LEVEL >= LogLevel.Level.INFO:
            print(f'MAE: {mae_avg:.4f}, MSE: {mse_avg:.4f}')
    return predictions, targets, mae_avg, mse_avg
