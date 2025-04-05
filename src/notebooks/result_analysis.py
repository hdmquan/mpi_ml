# %% Imports
import torch
import pytorch_lightning as pl
from src.models.fno import FNOPINN
from src.models.cnn import CNNPINN
from src.data.module import MPIDataModule
from src.utils.plotting import plot_layer, plot_long_cut
from src.utils import PATH

# %% Load data
datamodule = MPIDataModule(include_coordinates=False, batch_size=1, num_workers=4)
datamodule.setup()

# %% Load best model
# model = FNOPINN.load_from_checkpoint(PATH.CHECKPOINTS / "fno-best-model.ckpt")
model = CNNPINN.load_from_checkpoint(
    PATH.CHECKPOINTS / "cnn-best-model.ckpt"
)
model.eval()

# %% Get predictions
batch = next(iter(datamodule.train_dataloader()))
x, y = [b.to(model.device) for b in batch]
with torch.no_grad():
    y_pred = model(x)

# %% Visualization
# Ground truth
plot_layer(y.cpu().detach(), -2, save=True, x_range=[0, 1])  # Level 1

# Predictions
plot_layer(y_pred.cpu().detach(), -2, save=True, x_range=[0, 1])  # Level 1

# Ground truth longitude cuts
plot_long_cut(y.cpu().detach(), y.shape[-1] // 2, save=True, x_range=[0, 1])

# Predictions longitude cuts
plot_long_cut(y_pred.cpu().detach(), y_pred.shape[-1] // 2, save=True, x_range=[0, 1])

# %% Get predictions
batch = next(iter(datamodule.test_dataloader()))
x, y = [b.to(model.device) for b in batch]
with torch.no_grad():
    y_pred = model(x)

# %% Visualization
# Ground truth
plot_layer(y.cpu().detach(), 0, save=True, x_range=[0, 1])  # Level 1

# Predictions
plot_layer(y_pred.cpu().detach(), 0, save=True, x_range=[0, 1])  # Level 1

# Ground truth longitude cuts
plot_long_cut(y.cpu().detach(), y.shape[-1] // 2, save=True, x_range=[0, 1])

# Predictions longitude cuts
plot_long_cut(y_pred.cpu().detach(), y_pred.shape[-1] // 2, save=True, x_range=[0, 1])

# %%
