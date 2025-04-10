# %% Imports
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.models.cnn import CNNPINNStream
from src.data.module import MPIDataModule
from src.utils import PATH

# %% Load data
datamodule = MPIDataModule(include_coordinates=False, batch_size=1, num_workers=4)
datamodule.setup()

# %% Load and process first model
model1 = CNNPINNStream.load_from_checkpoint(
    PATH.CHECKPOINTS / "best-model-32_64_32-128_transport.ckpt",
    map_location=torch.device("cpu"),
)
model1.eval()

# Initialize arrays to store metrics for all instances
# Shape will be [6, 6, 48] (6 instances, 6 channels, 48 altitudes)
all_rmse1 = []
all_r2_1 = []

# Get test dataloader
test_loader = datamodule.test_dataloader()

# Process 6 different instances
for instance_idx in range(6):
    # Get a new instance each time
    batch = next(iter(test_loader))
    x, y, metadata = batch
    x = x.cpu()
    y = y.cpu()

    # Get prediction
    with torch.no_grad():
        y_pred = model1(x)

    # Calculate metrics for each channel and altitude
    # y shape: [1, 6, 50, 384, 576] -> [6, 48] after mean over spatial dimensions
    # We exclude the last 2 altitudes which are for deposition
    mse = ((y[:, :, :-2, :, :].numpy() - y_pred[:, :, :-2, :, :].numpy()) ** 2).mean(
        axis=(0, 3, 4)
    )
    rmse = np.sqrt(mse)

    # Calculate R²
    y_true = y[:, :, :-2, :, :].numpy()
    y_true_mean = y_true.mean(axis=(0, 3, 4), keepdims=True)
    total_ss = ((y_true - y_true_mean) ** 2).mean(axis=(0, 3, 4))
    r2 = 1 - (mse / total_ss)

    all_rmse1.append(rmse)
    all_r2_1.append(r2)

    # Clear memory
    del x, y, y_pred
    torch.cuda.empty_cache()

# Convert to numpy arrays with shape [6, 6, 48]
rmse_model1 = np.array(all_rmse1)
r2_model1 = np.array(all_r2_1)
del model1
torch.cuda.empty_cache()

# %% Load and process second model
model2 = CNNPINNStream.load_from_checkpoint(
    PATH.CHECKPOINTS / "best-model-32_64_32-128.ckpt",
    map_location=torch.device("cpu"),
)
model2.eval()

all_rmse2 = []
all_r2_2 = []

# Process 6 different instances again
for instance_idx in range(6):
    # Get a new instance each time
    batch = next(iter(test_loader))
    x, y, metadata = batch
    x = x.cpu()
    y = y.cpu()

    # Get prediction
    with torch.no_grad():
        y_pred = model2(x)

    # Calculate metrics for each channel and altitude
    mse = ((y[:, :, :-2, :, :].numpy() - y_pred[:, :, :-2, :, :].numpy()) ** 2).mean(
        axis=(0, 3, 4)
    )
    rmse = np.sqrt(mse)

    # Calculate R²
    y_true = y[:, :, :-2, :, :].numpy()
    y_true_mean = y_true.mean(axis=(0, 3, 4), keepdims=True)
    total_ss = ((y_true - y_true_mean) ** 2).mean(axis=(0, 3, 4))
    r2 = 1 - (mse / total_ss)

    all_rmse2.append(rmse)
    all_r2_2.append(r2)

    # Clear memory
    del x, y, y_pred
    torch.cuda.empty_cache()

# Convert to numpy arrays with shape [6, 6, 48]
rmse_model2 = np.array(all_rmse2)
r2_model2 = np.array(all_r2_2)
del model2
torch.cuda.empty_cache()

# %% Create RMSE plot
# Shape must be [6, 6, 48] (6 instances, 6 channels, 48 altitudes)
# 6 subplot (channels)
# 6 points line chart (instances)
# 48 altitudes for average and min/max for filled area and solid line
fig_rmse = make_subplots(
    rows=2, cols=3, subplot_titles=[f"Instance {i+1}" for i in range(6)]
)

for i in range(6):
    row = (i // 3) + 1
    col = (i % 3) + 1

    x = list(range(len(rmse_model1[i])))
    x_rev = x[::-1]

    # Add Model 1 filled area
    fig_rmse.add_trace(
        go.Scatter(
            x=x + x_rev,
            y=np.concatenate(
                [
                    rmse_model1[i].max() * np.ones_like(rmse_model1[i]),
                    rmse_model1[i].min() * np.ones_like(rmse_model1[i])[::-1],
                ]
            ),
            fill="toself",
            fillcolor="rgba(0, 0, 255, 0.2)",
            line_color="rgba(255, 255, 255, 0)",
            name="Model 1 Range" if i == 0 else None,
            showlegend=i == 0,
        ),
        row=row,
        col=col,
    )

    # Add Model 2 filled area
    fig_rmse.add_trace(
        go.Scatter(
            x=x + x_rev,
            y=np.concatenate(
                [
                    rmse_model2[i].max() * np.ones_like(rmse_model2[i]),
                    rmse_model2[i].min() * np.ones_like(rmse_model2[i])[::-1],
                ]
            ),
            fill="toself",
            fillcolor="rgba(255, 0, 0, 0.2)",
            line_color="rgba(255, 255, 255, 0)",
            name="Model 2 Range" if i == 0 else None,
            showlegend=i == 0,
        ),
        row=row,
        col=col,
    )

    # Add Model 1 line
    fig_rmse.add_trace(
        go.Scatter(
            x=x,
            y=rmse_model1[i],
            line_color="rgb(0, 0, 255)",
            name="Model 1" if i == 0 else None,
            showlegend=i == 0,
        ),
        row=row,
        col=col,
    )

    # Add Model 2 line
    fig_rmse.add_trace(
        go.Scatter(
            x=x,
            y=rmse_model2[i],
            line_color="rgb(255, 0, 0)",
            name="Model 2" if i == 0 else None,
            showlegend=i == 0,
        ),
        row=row,
        col=col,
    )

fig_rmse.update_traces(mode="lines")
fig_rmse.update_layout(
    title="RMSE by Altitude Level", height=800, width=1200, showlegend=True
)

# Update x and y axis labels
for i in range(1, 7):
    fig_rmse.update_xaxes(
        title_text="Altitude Level", row=(i - 1) // 3 + 1, col=(i - 1) % 3 + 1
    )
    fig_rmse.update_yaxes(title_text="RMSE", row=(i - 1) // 3 + 1, col=(i - 1) % 3 + 1)

fig_rmse.show()

# %% Create R² plot
fig_r2 = make_subplots(
    rows=2, cols=3, subplot_titles=[f"Instance {i+1}" for i in range(6)]
)

for i in range(6):
    row = (i // 3) + 1
    col = (i % 3) + 1

    x = list(range(len(r2_model1[i])))
    x_rev = x[::-1]

    # Add Model 1 filled area
    fig_r2.add_trace(
        go.Scatter(
            x=x + x_rev,
            y=np.concatenate(
                [
                    r2_model1[i].max() * np.ones_like(r2_model1[i]),
                    r2_model1[i].min() * np.ones_like(r2_model1[i])[::-1],
                ]
            ),
            fill="toself",
            fillcolor="rgba(0, 0, 255, 0.2)",
            line_color="rgba(255, 255, 255, 0)",
            name="Model 1 Range" if i == 0 else None,
            showlegend=i == 0,
        ),
        row=row,
        col=col,
    )

    # Add Model 2 filled area
    fig_r2.add_trace(
        go.Scatter(
            x=x + x_rev,
            y=np.concatenate(
                [
                    r2_model2[i].max() * np.ones_like(r2_model2[i]),
                    r2_model2[i].min() * np.ones_like(r2_model2[i])[::-1],
                ]
            ),
            fill="toself",
            fillcolor="rgba(255, 0, 0, 0.2)",
            line_color="rgba(255, 255, 255, 0)",
            name="Model 2 Range" if i == 0 else None,
            showlegend=i == 0,
        ),
        row=row,
        col=col,
    )

    # Add Model 1 line
    fig_r2.add_trace(
        go.Scatter(
            x=x,
            y=r2_model1[i],
            line_color="rgb(0, 0, 255)",
            name="Model 1" if i == 0 else None,
            showlegend=i == 0,
        ),
        row=row,
        col=col,
    )

    # Add Model 2 line
    fig_r2.add_trace(
        go.Scatter(
            x=x,
            y=r2_model2[i],
            line_color="rgb(255, 0, 0)",
            name="Model 2" if i == 0 else None,
            showlegend=i == 0,
        ),
        row=row,
        col=col,
    )

fig_r2.update_traces(mode="lines")
fig_r2.update_layout(
    title="R² by Altitude Level", height=800, width=1200, showlegend=True
)

# Update x and y axis labels
for i in range(1, 7):
    fig_r2.update_xaxes(
        title_text="Altitude Level", row=(i - 1) // 3 + 1, col=(i - 1) % 3 + 1
    )
    fig_r2.update_yaxes(title_text="R²", row=(i - 1) // 3 + 1, col=(i - 1) % 3 + 1)

fig_r2.show()

# %% Print average metrics
print("Model 1 Average Metrics:")
print(f"RMSE: {rmse_model1.mean():.6f}")
print(f"R²: {r2_model1.mean():.6f}")

print("\nModel 2 Average Metrics:")
print(f"RMSE: {rmse_model2.mean():.6f}")
print(f"R²: {r2_model2.mean():.6f}")
