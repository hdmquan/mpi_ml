# %% Imports
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import torch
from src.utils import PATH
from src.data.module import MPIDataModule
from src.utils.plotting import plot_layer, plot_long_cut


# %% Data
loader = MPIDataModule(batch_size=1)
loader.setup()

# %%
data = next(iter(loader.train_dataloader()))

# %%
X, Y = data

# %%
# [1, 6, 48, 384, 576]
print(X.shape)
# [1, 6, 48, 384, 576]
b, c, l, x, y = Y.shape

# %% Level-by-level inspection
lev = 48

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
