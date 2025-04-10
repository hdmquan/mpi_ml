# %% Imports
import torch
import plotly.graph_objects as go
import numpy as np
from src.models.cnn import CNNPINNStream
from src.data.module import MPIDataModule
from src.utils import PATH

# %% Load data
datamodule = MPIDataModule(include_coordinates=False, batch_size=1, num_workers=4)
datamodule.setup()

# %% Load best model
model = CNNPINNStream.load_from_checkpoint(
    PATH.CHECKPOINTS / "best-model-32_64_32-128_transport.ckpt",
    map_location=torch.device("cpu"),
)
model.eval()

# %% Get predictions
batch = next(iter(datamodule.train_dataloader()))

x, y, metadata = batch
x = x.cpu()
y = y.cpu()

with torch.no_grad():
    y_pred = model(x)

y = y[0, :, 0]
y_pred = y_pred[0, :, 0]

# Reshape tensors assuming 6 instances
num_instances = 6
y = y.reshape(num_instances, -1)
y_pred = y_pred.reshape(num_instances, -1)

print(f"Prediction range: {y_pred.min().item()} to {y_pred.max().item()}")
print(f"Ground truth range: {y.min().item()} to {y.max().item()}")

# %% Create subplot with prediction vs ground truth scatter plots for all instances
# Create a 2x3 subplot grid using make_subplots
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=3, subplot_titles=[f"Instance {i+1}" for i in range(num_instances)]
)

# Calculate overall min/max for consistent axes
all_min = min(y.min().item(), y_pred.min().item())
all_max = max(y.max().item(), y_pred.max().item())

# Metrics to store results
metrics = {"mse": [], "mae": [], "r2": []}

for instance in range(num_instances):
    # Calculate row and column for subplot
    row = instance // 3 + 1
    col = instance % 3 + 1

    # Flatten tensors for scatter plot
    y_true_flat = y[instance].flatten().numpy()
    y_pred_flat = y_pred[instance].flatten().numpy()

    # Add scatter plot for predictions
    fig.add_trace(
        go.Scattergl(
            x=y_true_flat,
            y=y_pred_flat,
            mode="markers",
            marker=dict(color="rgba(31, 119, 180, 0.3)", size=2),
            name=f"Instance {instance+1}",
            showlegend=(instance == 0),  # Only show legend for first instance
        ),
        row=row,
        col=col,
    )

    # Add diagonal line (perfect prediction)
    fig.add_trace(
        go.Scatter(
            x=[all_min, all_max],
            y=[all_min, all_max],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Perfect prediction",
            showlegend=(instance == 0),  # Only show legend for first instance
        ),
        row=row,
        col=col,
    )

    # Calculate error metrics for this instance
    mse = ((y_true_flat - y_pred_flat) ** 2).mean()
    mae = np.abs(y_true_flat - y_pred_flat).mean()
    r2 = 1 - (
        ((y_true_flat - y_pred_flat) ** 2).sum()
        / ((y_true_flat - y_true_flat.mean()) ** 2).sum()
    )

    metrics["mse"].append(mse)
    metrics["mae"].append(mae)
    metrics["r2"].append(r2)

    # Add metrics as annotations
    fig.add_annotation(
        text=f"MSE: {mse:.4f}<br>MAE: {mae:.4f}<br>R²: {r2:.4f}",
        xref=f"x{instance+1}",
        yref=f"y{instance+1}",
        x=0.05,
        y=0.95,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        font=dict(size=10),
    )

# Update layout
fig.update_layout(
    title="Model Predictions vs Ground Truth",
    height=800,
    width=1200,
    template="plotly_white",
)

# Update all axes to be the same
for i in range(1, num_instances + 1):
    fig.update_xaxes(
        title_text="Ground Truth",
        range=[all_min, all_max],
        row=i // 3 + 1,
        col=i % 3 + 1,
    )
    fig.update_yaxes(
        title_text="Predictions",
        range=[all_min, all_max],
        row=i // 3 + 1,
        col=i % 3 + 1,
    )

fig.show()

# Print average metrics
print(f"Average MSE: {np.mean(metrics['mse']):.6f}")
print(f"Average MAE: {np.mean(metrics['mae']):.6f}")
print(f"Average R²: {np.mean(metrics['r2']):.6f}")

# %%
