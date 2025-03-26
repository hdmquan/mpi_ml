# %%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from plotly.subplots import make_subplots
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from loguru import logger

from src.data.module import MPIDataModule

# %% Set up data loading

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


# %%
def extract_features_labels(batch):
    inputs, (mmr, _, _), _ = batch
    # logger.debug(f"inputs: {inputs.shape} | mmr: {mmr.shape}")
    # inputs: torch.Size([1, 8, 48, 384, 576]) | mmr: torch.Size([1, 6, 48, 384, 576])
    # batch, channels, vertical, height, width
    inputs = inputs.numpy()
    inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], -1)
    inputs = inputs.transpose(0, 2, 1)
    inputs = inputs.reshape(-1, inputs.shape[2])

    mmr = mmr.numpy()
    mmr = mmr.reshape(mmr.shape[0], mmr.shape[1], -1)
    mmr = mmr.transpose(0, 2, 1)
    mmr = mmr.reshape(-1, mmr.shape[2])

    logger.debug(f"inputs: {inputs.shape} | mmr: {mmr.shape}")
    logger.debug(f"inputs: {inputs[:5]} | mmr: {mmr[:5]}")

    logger.debug(f"inputs range: {inputs.min()} - {inputs.max()}")
    logger.debug(f"mmr range: {mmr.min()} - {mmr.max()}")

    return inputs, mmr


# %%
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

            y_pred = model.predict(X_train)
            mse = mean_squared_error(y_train[:, i], y_pred)
            epoch_losses[i].append(mse)

    for i in range(6):
        if epoch_losses[i]:
            avg_loss = np.mean(epoch_losses[i])
            training_losses[i].append(avg_loss)
        else:
            training_losses[i].append(0)


# %%
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

# %%
y_true, y_pred = [], []
batch = next(iter(test_loader))
X, y = extract_features_labels(batch)
X = scaler.transform(X)

y_true = np.array(y)
y_pred = np.column_stack([model.predict(X) for model in models])

# %%
print(y_true.shape)
print(y_pred.shape)
y_true = y_true.reshape(48, 384, 576, 6)
y_pred = y_pred.reshape(48, 384, 576, 6)

# %%
subplots = make_subplots(rows=6, cols=2, subplot_titles=[f"Output {i}" for i in range(6)] * 2)

for i in range(6):
    # True values heatmap
    subplots.add_trace(
        go.Heatmap(z=y_true[0, :, :, i], colorscale="Viridis", colorbar=dict(title="True")),
        row=i + 1, col=1
    )
    
    # Predicted values heatmap
    subplots.add_trace(
        go.Heatmap(z=y_pred[0, :, :, i], colorscale="Viridis", colorbar=dict(title="Predicted")),
        row=i + 1, col=2
    )

subplots.update_layout(
    title="True vs. Predicted Values",
    height=1500,  # Adjust height for better spacing
    width=1000,
    showlegend=False,
)

subplots.show()

# %%
