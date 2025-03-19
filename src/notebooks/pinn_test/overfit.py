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

# Add the project root to the path
module_path = str(Path(__file__).parents[3])
if module_path not in sys.path:
    sys.path.append(module_path)

from src.models.fno.pinn import PINNModel
from src.data.datamodule import MicroplasticsDataModule

# %% Config
SEED = 42
pl.seed_everything(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Path to dataset
DATA_DIR = Path("data/processed/")
BATCH_SIZE = 1
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Load a single example from the dataset
print(f"Loading dataset from {DATA_DIR}")
datamodule = MicroplasticsDataModule(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=False,  # We want to get the same first sample every time
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
        self.model = model
        self.x = x
        self.targets = targets
        self.cell_area = cell_area
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Storage for metrics
        self.epochs = []
        self.losses = {}
        self.gradients = {}

    def train(self):
        self.model.to(DEVICE)
        self.x = self.x.to(DEVICE)
        self.targets = [t.to(DEVICE) for t in self.targets]
        self.cell_area = self.cell_area.to(DEVICE)

        for epoch in range(self.num_epochs):
            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            predictions = self.model(self.x)

            # Compute losses
            losses = self.model.compute_loss(
                self.x, predictions, self.targets, self.cell_area
            )

            # Total loss (data loss + physics loss)
            data_loss = losses["mmr"] + losses["deposition"]
            physics_loss = losses["total_physics"]
            total_loss = data_loss + physics_loss
            losses["data_loss"] = data_loss
            losses["total_loss"] = total_loss

            # Backward pass and optimize
            total_loss.backward()

            # Record losses
            self.epochs.append(epoch)
            for loss_name, loss_value in losses.items():
                if loss_name not in self.losses:
                    self.losses[loss_name] = []
                self.losses[loss_name].append(loss_value.item())

            # Record parameter gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if name not in self.gradients:
                        self.gradients[name] = []
                    # Use mean absolute gradient for each parameter
                    self.gradients[name].append(param.grad.abs().mean().item())

            # Step optimizer
            self.optimizer.step()

            # Print progress
            if epoch % 1 == 0:
                print(
                    f"Epoch {epoch}: Total Loss = {total_loss.item():.6f}, "
                    f"Data Loss = {data_loss.item():.6f}, "
                    f"Physics Loss = {physics_loss.item():.6f}"
                )

        print("Training complete!")
        return self.losses["total_loss"][-1]


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
fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=["Training Losses", "Parameter Gradients"],
    vertical_spacing=0.15,
)

# Plot all losses
for loss_name, loss_values in trainer.losses.items():
    fig.add_trace(
        go.Scatter(x=trainer.epochs, y=loss_values, name=loss_name, mode="lines"),
        row=1,
        col=1,
    )

# Plot gradients - select a subset if there are too many parameters
gradient_names = list(trainer.gradients.keys())
if len(gradient_names) > 10:
    # Select a representative sample of gradients
    gradient_names = gradient_names[:10]

for name in gradient_names:
    fig.add_trace(
        go.Scatter(
            x=trainer.epochs,
            y=trainer.gradients[name],
            name=f"grad_{name}",
            mode="lines",
        ),
        row=2,
        col=1,
    )

# Update layout
fig.update_layout(
    height=800,
    width=1000,
    title_text="Overfitting Test Results",
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

fig.update_xaxes(title_text="Epoch", row=1, col=1)
fig.update_xaxes(title_text="Epoch", row=2, col=1)
fig.update_yaxes(title_text="Loss Value", row=1, col=1)
fig.update_yaxes(title_text="Gradient Magnitude", row=2, col=1)

# Use log scale for losses
fig.update_yaxes(type="log", row=1, col=1)

# Save the figure
fig.write_html("overfit_results.html")
fig.show()

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

# %% Evaluate if the model can fit this data point
# Define a threshold for what we consider "minimal" loss
LOSS_THRESHOLD = 1e-2
CAN_OVERFIT = final_loss < LOSS_THRESHOLD

print(f"Final loss: {final_loss:.6f}")
print(f"Loss threshold: {LOSS_THRESHOLD}")
print(f"Can the model overfit a single example? {'YES' if CAN_OVERFIT else 'NO'}")

assert CAN_OVERFIT, "Model failed to overfit a single training example"


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
