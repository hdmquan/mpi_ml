# %% Imports
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path

from src.models.fno.pinn import PINNModel
from src.data.module import MPIDataModule
from src.utils import set_seed

# Set random seeds for reproducibility
set_seed()

# %% Config
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Path to dataset
DATA_DIR = Path("data/processed/")
BATCH_SIZE = 1
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Load a single example from the dataset
print(f"Loading dataset from {DATA_DIR}")
datamodule = MPIDataModule(
    batch_size=BATCH_SIZE,
    num_workers=0,
)
datamodule.setup()

# Get the first sample from the training set
train_dataloader = datamodule.train_dataloader()
x, targets, cell_area = next(iter(train_dataloader))

print(f"Input shape: {x.shape}")
print(f"Target shapes: {[t.shape for t in targets]}")
print(f"Cell area shape: {cell_area.shape}")


# %% Define a custom trainer class to capture all metrics during training
class OverfitTrainer:
    def __init__(self, model, x, targets, cell_area, num_epochs, learning_rate):
        self.model = model.to(DEVICE)
        self.x = x.to(DEVICE)
        self.targets = [t.to(DEVICE) for t in targets]
        self.cell_area = cell_area.to(DEVICE)
        self.num_epochs = num_epochs
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Storage for metrics
        self.losses = {"total_loss": [], "data_loss": [], "physics_loss": []}

    def train(self):
        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()

            # Forward pass
            predictions = self.model(self.x)
            losses = self.model.compute_loss(
                self.x, predictions, self.targets, self.cell_area
            )

            # Calculate losses
            data_loss = losses["mmr"] + losses["deposition"]
            physics_loss = losses["total_physics"]
            total_loss = data_loss + physics_loss

            # Backward pass
            total_loss.backward()
            self.optimizer.step()

            # Store losses
            self.losses["total_loss"].append(total_loss.item())
            self.losses["data_loss"].append(data_loss.item())
            self.losses["physics_loss"].append(physics_loss.item())

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss.item():.6f}")

        return self.losses["total_loss"][-1]


def plot_training_curves(trainer):
    fig = go.Figure()

    for name, values in trainer.losses.items():
        fig.add_trace(go.Scatter(y=values, name=name, mode="lines"))

    fig.update_layout(
        title="Training Losses",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        yaxis_type="log",
    )
    return fig


# %% Create and train the model
model = PINNModel(
    in_channels=x.shape[1],
    out_channels=targets[0].shape[1] * 3,  # MMR, DryDep, WetDep for each size bin
    learning_rate=LEARNING_RATE,
    physics_weight=0.1,
    boundary_weight=0.1,
    conservation_weight=0.1,
    deposition_weight=0.1,
)

trainer = OverfitTrainer(
    model=model,
    x=x,
    targets=targets,
    cell_area=cell_area,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
)

final_loss = trainer.train()

# %% Plot losses
fig = plot_training_curves(trainer)
fig.write_html("overfit_results.html")
fig.show()

# %% Check if model successfully overfit
LOSS_THRESHOLD = 1e-2
print(f"\nFinal loss: {final_loss:.6f}")
print(f"Can the model overfit? {'YES' if final_loss < LOSS_THRESHOLD else 'NO'}")

# %% Test model predictions on the training sample
model.eval()
with torch.no_grad():
    mmr_pred, dry_dep_pred, wet_dep_pred = model(x.to(DEVICE))
    mmr_true, dry_dep_true, wet_dep_true = targets

    # Calculate RMSE for each output
    mmr_rmse = torch.sqrt(((mmr_pred - mmr_true.to(DEVICE)) ** 2).mean()).item()
    dry_dep_rmse = torch.sqrt(
        ((dry_dep_pred - dry_dep_true.to(DEVICE)) ** 2).mean()
    ).item()
    wet_dep_rmse = torch.sqrt(
        ((wet_dep_pred - wet_dep_true.to(DEVICE)) ** 2).mean()
    ).item()

    print(f"RMSE for MMR: {mmr_rmse:.6f}")
    print(f"RMSE for Dry Deposition: {dry_dep_rmse:.6f}")
    print(f"RMSE for Wet Deposition: {wet_dep_rmse:.6f}")


# %% Visualization of predicted vs actual values
# Plot first size bin for visualization
def plot_comparison(pred, true, title):
    fig = make_subplots(
        rows=1, cols=3, subplot_titles=["True", "Predicted", "Difference"]
    )

    # Get data for first particle size (index 0) and first batch item (index 0)
    true_data = true[0, 0].cpu().numpy()
    pred_data = pred[0, 0].cpu().numpy()
    diff_data = pred_data - true_data

    # Set same color scale for true and predicted
    vmin = min(true_data.min(), pred_data.min())
    vmax = max(true_data.max(), pred_data.max())

    # True values
    fig.add_trace(
        go.Heatmap(z=true_data, colorscale="Viridis", zmin=vmin, zmax=vmax),
        row=1,
        col=1,
    )

    # Predicted values
    fig.add_trace(
        go.Heatmap(z=pred_data, colorscale="Viridis", zmin=vmin, zmax=vmax),
        row=1,
        col=2,
    )

    # Difference
    fig.add_trace(go.Heatmap(z=diff_data, colorscale="RdBu", zmid=0), row=1, col=3)

    fig.update_layout(title_text=title, height=400, width=1200)

    return fig


# Plot comparisons for each output
mmr_fig = plot_comparison(mmr_pred, mmr_true.to(DEVICE), "MMR: True vs Predicted")
dry_dep_fig = plot_comparison(
    dry_dep_pred, dry_dep_true.to(DEVICE), "Dry Deposition: True vs Predicted"
)
wet_dep_fig = plot_comparison(
    wet_dep_pred, wet_dep_true.to(DEVICE), "Wet Deposition: True vs Predicted"
)

mmr_fig.show()
dry_dep_fig.show()
wet_dep_fig.show()

print("Overfitting test completed!")
