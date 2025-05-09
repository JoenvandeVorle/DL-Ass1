# visualize.py
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os


def visualize_laser_data(mat_file: str, output_dir: str = "plots") -> None:
    # Load the .mat file
    mat_data = scipy.io.loadmat(mat_file)

    # Check available variable names
    print(mat_data.keys())

    # Load and squeeze the actual data
    data = mat_data['Xtrain']
    laser_data = np.squeeze(data)

    # Plot the data as points
    plt.figure(figsize=(10, 4))
    # plt.scatter(range(len(laser_data)), laser_data, s=10, c='blue', marker='o')  # 's' is size
    plt.plot(laser_data, label='Laser Data', color='blue', linewidth=2)
    plt.title('Laser Measurements')
    plt.xlabel('Time Step')
    plt.ylabel('Distance (or relevant unit)')
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filename = "laser_plot.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()


def visualize_training(csv_path: str, output_dir: str = "plots") -> None:
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    if not {"epoch", "train_losses", "val_losses"}.issubset(df.columns):
        print("CSV does not contain required columns.")
        sys.exit(1)

    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_losses"], label="Train Loss", color="blue", linewidth=2)
    plt.plot(df["epoch"], df["val_losses"], label="Validation Loss", color="orange", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(csv_path).replace(".csv", "_loss_plot.png")
    plt.savefig(os.path.join(output_dir, filename))
    # plt.show()

def visualize_predictions(predictions: list[float], targets: list[float], filename: str = "predictions_vs_targets.png", output_dir: str = "plots") -> None:

    plt.figure(figsize=(10, 6))
    plt.plot(predictions, label="Predictions", color="blue", linewidth=2)
    plt.plot(targets, label="Targets", color="orange", linewidth=2)
    # plt.scatter(range(len(predictions)), predictions, label="Predictions", color="blue", s=10)
    # plt.scatter(range(len(targets)), targets, label="Targets", color="orange", s=10)
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.title("Predictions vs Targets")
    plt.legend()
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename))
    # plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize.py <results_csv_path>")
        sys.exit(1)

    # visualize_training(sys.argv[1])
    visualize_laser_data(sys.argv[1])
