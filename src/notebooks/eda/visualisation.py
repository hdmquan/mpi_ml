# %% Imports
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import torch
from src.utils import PATH
from src.data.module import MPIDataModule


def plot_layer(x, lev, num_channels=6, titles=None):
    rows = (num_channels + 1) // 2
    if titles is None:
        titles = [f"Bin {i+1}" for i in range(num_channels)]

    fig = make_subplots(
        rows=rows,
        cols=2,
        subplot_titles=titles,
    )

    for i in range(num_channels):
        # :D
        row = i // 2 + 1
        col = i % 2 + 1

        fig.add_trace(
            go.Heatmap(
                z=x[0, i, lev, :, :],
                colorscale="RdBu",
                reversescale=True,
                # zmin=-1,
                # zmax=1,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        height=800,
    )

    fig.show()


def plot_long_cut(x, long, num_channels=6, titles=None):
    rows = (num_channels + 1) // 2
    if titles is None:
        titles = [f"Bin {i+1}" for i in range(num_channels)]

    fig = make_subplots(rows=rows, cols=2, subplot_titles=titles)

    for i in range(num_channels):
        # :D
        row = i // 2 + 1
        col = i % 2 + 1

        fig.add_trace(
            go.Heatmap(
                z=x[0, i, :, :, long],
                colorscale="RdBu",
                reversescale=True,
                # zmin=-1,
                # zmax=1,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        height=800,
    )

    fig.show()


# %% Data
loader = MPIDataModule(batch_size=1)
loader.setup()

# %%
data = next(iter(loader.train_dataloader()))

# %%
X = data[0][:, :-2, :, :, :]
Y = data[1][0]

# %%
# [1, 6, 48, 384, 576]
print(X.shape)
# [1, 6, 48, 384, 576]
b, c, l, x, y = Y.shape

# %% Level-by-level inspection
lev = 1

plot_layer(Y, lev - 1)

# %% Longitude cut
plot_long_cut(Y, y // 2)

# %%
titles = [
    "Pressure",
    "Wind (North)",
    "Wind (East)",
    "Temperature",
    "Humidity",
    "Tropospheric pressure",
]
lev = 1
plot_layer(X, lev - 1, titles=titles)

# %%
plot_long_cut(X, x // 2, titles=titles)

# %%
