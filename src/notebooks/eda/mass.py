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
min_val = -0.8929
max_val = 500.0

sums = []

for batch in loader.train_dataloader():
    Y = batch[1]
    Y_normalized = (Y - min_val) / (max_val - min_val)
    sums.append(Y_normalized.sum())

# %%
fig = px.line(sums, labels={"x": "Month", "y": "Sum"}, title="Monthly Sums")
fig.update_xaxes(
    tickvals=np.arange(12),
    ticktext=[
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ],
)
# Adjusting the y-axis to use a linear scale instead of log
fig.update_yaxes(range=[0, max(sums)], type="linear")
fig.show()
# %%
