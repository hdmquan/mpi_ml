# %% Import required libraries
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from loguru import logger
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import matplotlib.pyplot as plt

# %% Set up data loading
from src.data.module import MPIDataModule

datamodule = MPIDataModule(batch_size=1, shuffle=True)
datamodule.setup()

train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()

# %% Define model parameters
scaler = StandardScaler()

base_sgd = SGDRegressor(
    loss="huber",
    learning_rate="adaptive",
    eta0=0.01,
    penalty="l2",
    alpha=0.0001,
    max_iter=1000,
    tol=1e-3,
    warm_start=True,  # On for partial_fit
    random_state=37,
)

models = [SGDRegressor(**base_sgd.get_params()) for _ in range(6)]


# %% Define data processing functions
def extract_features_labels(batch):
    inputs, (mmr, _, _), _ = batch
    # logger.debug(f"inputs: {inputs.shape} | mmr: {mmr.shape}")
    # inputs: torch.Size([1, 8, 48, 384, 576]) | mmr: torch.Size([1, 6, 48, 384, 576])
    inputs = inputs.numpy()
    inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], -1)
    inputs = inputs.transpose(0, 2, 1)
    inputs = inputs.reshape(-1, inputs.shape[2])

    mmr = mmr.numpy()
    mmr = mmr.reshape(mmr.shape[0], mmr.shape[1], -1)
    mmr = mmr.transpose(0, 2, 1)
    mmr = mmr.reshape(-1, mmr.shape[2])

    return inputs, mmr


# %% Initialize scaler with first batch
first_batch = next(iter(train_loader))
X_init, _ = extract_features_labels(first_batch)
scaler.fit(X_init)

# %% Train models
training_losses = {i: [] for i in range(6)}
epoch_numbers = []

for epoch in range(5):
    epoch_numbers.append(epoch)
    epoch_losses = {i: [] for i in range(6)}

    for batch in train_loader:
        X_train, y_train = extract_features_labels(batch)
        X_train = scaler.transform(X_train)

        logger.debug(f"X: {X_train.shape} | y: {y_train.shape}")

        for i, model in enumerate(models):
            model.partial_fit(X_train, y_train[:, i])
            # Calculate loss for visualization
            y_pred = model.predict(X_train)
            mse = mean_squared_error(y_train[:, i], y_pred)
            epoch_losses[i].append(mse)

    # Average losses for the epoch
    for i in range(6):
        if epoch_losses[i]:
            avg_loss = np.mean(epoch_losses[i])
            training_losses[i].append(avg_loss)
        else:
            training_losses[i].append(0)


# %% Define evaluation function
def evaluate(loader, name):
    y_true, y_pred = [], []
    for batch in loader:
        X, y = extract_features_labels(batch)
        X = scaler.transform(X)  # Standardize features

        batch_preds = np.column_stack([model.predict(X) for model in models])
        y_true.append(y)
        y_pred.append(batch_preds)

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    mse_per_output = mean_squared_error(y_true, y_pred, multioutput="raw_values")
    avg_mse = mean_squared_error(y_true, y_pred)
    print(f"{name} Average MSE: {avg_mse:.4f}")
    for i, mse in enumerate(mse_per_output):
        print(f"{name} MSE (output {i}): {mse:.4f}")

    return avg_mse, mse_per_output


# %% Evaluate on validation and test sets
val_avg_mse, val_mse_per_output = evaluate(val_loader, "Validation")
test_avg_mse, test_mse_per_output = evaluate(test_loader, "Test")

# %% Visualize training loss over epochs
fig = go.Figure()
for i in range(6):
    fig.add_trace(
        go.Scatter(
            x=epoch_numbers,
            y=training_losses[i],
            mode="lines+markers",
            name=f"Output {i}",
        )
    )

fig.update_layout(
    title="Training Loss per Output over Epochs",
    xaxis_title="Epoch",
    yaxis_title="Mean Squared Error",
    legend_title="Output",
)
fig.show()

# %% Visualize validation and test MSE per output
output_names = [f"Output {i}" for i in range(6)]
results_df = pd.DataFrame(
    {
        "Output": output_names + ["Average"],
        "Validation MSE": list(val_mse_per_output) + [val_avg_mse],
        "Test MSE": list(test_mse_per_output) + [test_avg_mse],
    }
)

fig = px.bar(
    results_df,
    x="Output",
    y=["Validation MSE", "Test MSE"],
    barmode="group",
    title="Evaluation Results: MSE by Output",
)
fig.update_layout(
    xaxis_title="Output", yaxis_title="Mean Squared Error", legend_title="Dataset"
)
fig.show()

# %% Visualize prediction distribution (for the first 1000 samples from test set)
sample_size = 1000
sample_batch = next(iter(test_loader))
X_sample, y_true_sample = extract_features_labels(sample_batch)
X_sample = scaler.transform(X_sample)
y_pred_sample = np.column_stack([model.predict(X_sample) for model in models])

# Limit to the first 1000 samples if needed
if X_sample.shape[0] > sample_size:
    X_sample = X_sample[:sample_size]
    y_true_sample = y_true_sample[:sample_size]
    y_pred_sample = y_pred_sample[:sample_size]

# Create plot for prediction vs. actual
fig = make_subplots(rows=2, cols=3, subplot_titles=[f"Output {i}" for i in range(6)])

for i in range(6):
    row, col = i // 3 + 1, i % 3 + 1
    fig.add_trace(
        go.Scatter(
            x=y_true_sample[:, i],
            y=y_pred_sample[:, i],
            mode="markers",
            marker=dict(size=5, opacity=0.6),
            name=f"Output {i}",
        ),
        row=row,
        col=col,
    )

    # Add 45-degree reference line (perfect predictions)
    min_val = min(y_true_sample[:, i].min(), y_pred_sample[:, i].min())
    max_val = max(y_true_sample[:, i].max(), y_pred_sample[:, i].max())

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            showlegend=False,
        ),
        row=row,
        col=col,
    )

fig.update_layout(
    title="Predicted vs. Actual Values for Test Set",
    height=800,
    width=1000,
    showlegend=False,
)

for i in range(6):
    row, col = i // 3 + 1, i % 3 + 1
    fig.update_xaxes(title_text="Actual Value", row=row, col=col)
    fig.update_yaxes(title_text="Predicted Value", row=row, col=col)

fig.show()

# %% Visualize output as 2D images
# Get a single test sample
test_batch = next(iter(test_loader))
_, (mmr_true, _, _), _ = test_batch

# Get raw inputs for this batch
inputs_raw, y_true_sample = extract_features_labels(test_batch)
inputs_scaled = scaler.transform(inputs_raw)

# Generate predictions
predictions = np.column_stack([model.predict(inputs_scaled) for model in models])

# Reshape the predictions and ground truth back to original image dimensions
batch_size, channels, time_steps, height, width = mmr_true.shape

# Get the first sample (batch_idx=0) and the first time step (time_idx=0)
sample_idx = 0
time_idx = 0

# Reshape predictions and ground truth back to 2D images
pred_images = []
true_images = []

for channel in range(channels):
    # Extract ground truth
    true_2d = mmr_true[sample_idx, channel, time_idx].numpy()
    true_images.append(true_2d)

    # For predictions, we need to reshape the flattened predictions back to 2D
    # First get predictions for this channel
    channel_preds = predictions[:, channel]

    # Reshape to match the original image dimensions
    pred_3d = channel_preds.reshape(48, height, width)
    pred_2d = pred_3d[0, ...]
    pred_images.append(pred_2d)

# Create a subplot grid for the 6 channels (both pred and true)
fig = make_subplots(
    rows=2,
    cols=6,
    subplot_titles=[f"Channel {i} (Pred)" for i in range(6)]
    + [f"Channel {i} (True)" for i in range(6)],
)

# Add prediction images
for i in range(6):
    fig.add_trace(
        go.Heatmap(z=pred_images[i], colorscale="Viridis", showscale=False),
        row=1,
        col=i + 1,
    )

# Add ground truth images
for i in range(6):
    fig.add_trace(
        go.Heatmap(z=true_images[i], colorscale="Viridis", showscale=False),
        row=2,
        col=i + 1,
    )

fig.update_layout(
    title_text="Comparison of Predicted vs. True Images for Each Channel",
    height=600,
    width=1200,
)

fig.show()

# %% Alternative visualization with shared colorscale
# Find global min and max for consistent colorscale
global_min = min(
    min(np.min(img) for img in pred_images), min(np.min(img) for img in true_images)
)
global_max = max(
    max(np.max(img) for img in pred_images), max(np.max(img) for img in true_images)
)

# Create two rows of plots
fig, axes = plt.subplots(2, 6, figsize=(20, 8))
fig.suptitle("Predicted vs True Output Channels", fontsize=16)

# Plot predictions on top row
for i in range(6):
    im = axes[0, i].imshow(
        pred_images[i], vmin=global_min, vmax=global_max, cmap="viridis"
    )
    axes[0, i].set_title(f"Channel {i} (Predicted)")
    axes[0, i].axis("off")

# Plot ground truth on bottom row
for i in range(6):
    im = axes[1, i].imshow(
        true_images[i], vmin=global_min, vmax=global_max, cmap="viridis"
    )
    axes[1, i].set_title(f"Channel {i} (Ground Truth)")
    axes[1, i].axis("off")

# Add a colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
cbar.set_label("Value")

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()
