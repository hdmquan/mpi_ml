# %% Imports
from random import betavariate
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from src.models.fno import FNOPINN
from src.models.cnn import CNNPINN, CNNPINNStream
from src.data.module import MPIDataModule
from src.utils.plotting import plot_layer, plot_long_cut
from src.utils import set_seed, PATH

set_seed()

# %% Data
datamodule = MPIDataModule(
    latitude_split=1, longitude_split=1, batch_size=1, num_workers=4
)

# %% Model setup
# model = FNOPINN(use_physics_loss=False)
# model = CNNPINN(in_channels=11, output_altitude_dim=48 + 2, use_physics_loss=False)
model = CNNPINNStream(scaled_loss=False, use_physics_loss=True, use_mass_conservation_loss=False)

# %% Training configuration
checkpoint_dir = PATH.CHECKPOINTS
checkpoint_path = checkpoint_dir / "best-model.ckpt"

# Create checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename="best-model",
    monitor="val_total",
    mode="min",
    save_top_k=1,
)

# Training configuration
trainer = pl.Trainer(
    max_epochs=100,
    accelerator="cuda" if torch.cuda.is_available() else "cpu",
    callbacks=[
        checkpoint_callback,
        pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_total", patience=5, mode="min", min_delta=0.0005
        ),
    ],
    enable_progress_bar=True,
    num_sanity_val_steps=0,
)

# Load from checkpoint if it exists
if checkpoint_path.exists():
    print(f"Resuming training from checkpoint: {checkpoint_path}")
    trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_path)
else:
    print("Starting training from scratch")
    trainer.fit(model, datamodule=datamodule)

# %% Testing
test_results = trainer.test(model, datamodule=datamodule)

# %% Visualization
batch = next(iter(datamodule.test_dataloader()))

x, y, metadata = batch
x = x.to(model.device)
y = y.to(model.device)

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
