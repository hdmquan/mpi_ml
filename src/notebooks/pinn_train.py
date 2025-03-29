# %% Imports
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from src.models.fno import FNOPINN
from src.data.module import MPIDataModule
from src.utils.plotting import plot_layer, plot_long_cut

# %% Data
datamodule = MPIDataModule(batch_size=1, num_workers=4)

# %% Model setup
model = FNOPINN(use_physics_loss=True)

# %% Training configuration
trainer = pl.Trainer(
    max_epochs=10,
    accelerator="cuda" if torch.cuda.is_available() else "cpu",
    callbacks=[
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="best-model",
            monitor="val_total",
            mode="min",
            save_top_k=1,
        )
    ],
    enable_progress_bar=True,
)

# %% Training
trainer.fit(model, datamodule=datamodule)

# %% Testing
test_results = trainer.test(model, datamodule=datamodule)

# %% Visualization
batch = next(iter(datamodule.test_dataloader()))
x, y, cell_area = [b.to(model.device) for b in batch]
y_pred = model(x)

# Ground truth
plot_layer(y.cpu().detach(), 0, save=True)  # Level 1

# Predictions
plot_layer(y_pred.cpu().detach(), 0, save=True)  # Level 1

# Ground truth longitude cuts
plot_long_cut(y.cpu().detach(), y.shape[-1] // 2, save=True)

# Predictions longitude cuts
plot_long_cut(y_pred.cpu().detach(), y_pred.shape[-1] // 2, save=True)

# %%
