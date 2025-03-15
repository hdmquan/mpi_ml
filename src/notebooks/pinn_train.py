import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

from src.data.module import MicroplasticDataModule
from src.models.fno.pinn import PINNModel
from src.utils import set_seed, PATH

# Set random seed for reproducibility
set_seed()

MAX_EPOCHS = 100

# Create directories for saving results
results_dir = PATH.RESULTS / "pinn_training"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(results_dir / "plots", exist_ok=True)
os.makedirs(results_dir / "gradients", exist_ok=True)
os.makedirs(results_dir / "predictions", exist_ok=True)

# Lists to store metrics for plotting
train_losses = []
val_losses = []
val_epochs = []  # Track which epochs have validation data
physics_losses = []
data_losses = []
epochs = []

# Gradient tracking
param_names = []
param_grads = {}

# Configure data module
data_module = MicroplasticDataModule(
    batch_size=4,
    num_workers=4,
    use_normalized=True,
    surface_level_only=True,
    include_derivatives=False,
    include_coordinates=True,
)

# Prepare the data
data_module.prepare_data()
data_module.setup()

# Get input and output dimensions from a sample batch
sample_batch = next(iter(data_module.train_dataloader()))
in_channels = sample_batch["inputs"].shape[1]
out_channels = sample_batch["targets"].shape[1]

print(f"Input channels: {in_channels}")
print(f"Output channels: {out_channels}")


# Custom callback to track gradients and losses
class MetricsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.first_batch_processed = False

    def on_train_epoch_end(self, trainer, pl_module):
        # Record losses
        train_losses.append(trainer.callback_metrics.get("train_total_loss", 0).item())
        physics_losses.append(
            trainer.callback_metrics.get("train_conservation_loss", 0).item()
        )
        data_losses.append(trainer.callback_metrics.get("train_data_loss", 0).item())
        epochs.append(trainer.current_epoch)

        # Save gradients for key parameters
        if not param_names:
            for name, param in pl_module.named_parameters():
                if param.requires_grad:
                    param_names.append(name)
                    param_grads[name] = []

        for name, param in pl_module.named_parameters():
            if param.requires_grad and param.grad is not None:
                param_grads[name].append(param.grad.norm().item())

    def on_validation_epoch_end(self, trainer, pl_module):
        # Record validation loss and the epoch it occurred
        val_losses.append(trainer.callback_metrics.get("val_total_loss", 0).item())
        val_epochs.append(trainer.current_epoch)

        # Save prediction visualization after validation
        if trainer.current_epoch % 10 == 0:
            self.visualize_predictions(pl_module, trainer.current_epoch)

    def visualize_predictions(self, pl_module, epoch):
        # Get a sample batch
        batch = next(iter(data_module.val_dataloader()))

        # Move to the same device as the model
        x = batch["inputs"].to(pl_module.device)
        coords = batch["coords"].to(pl_module.device)
        y_true = batch["targets"].to(pl_module.device)

        # Generate predictions
        with torch.no_grad():
            pl_module.eval()
            y_pred = pl_module(x, coords)
            pl_module.train()

        # Create visualization for the first sample in batch
        self.plot_prediction(x[0], y_true[0], y_pred[0], coords[0], epoch)

    def plot_prediction(self, x_sample, y_true, y_pred, coords, epoch):
        # Create a figure with subplots for each plastic size bin
        fig, axes = plt.subplots(out_channels, 3, figsize=(15, 4 * out_channels))

        # Get lat/lon for plotting
        lon = coords[0].cpu().numpy()
        lat = coords[1].cpu().numpy()

        # Handle the case where out_channels is 1
        if out_channels == 1:
            axes = [axes]  # Make axes iterable

        for i in range(out_channels):
            # Plot true concentration
            im0 = axes[i][0].pcolormesh(
                lon, lat, y_true[i].cpu().numpy(), shading="auto", cmap="viridis"
            )
            axes[i][0].set_title(f"True Concentration (Bin {i+1})")
            plt.colorbar(im0, ax=axes[i][0])

            # Plot predicted concentration
            im1 = axes[i][1].pcolormesh(
                lon, lat, y_pred[i].cpu().numpy(), shading="auto", cmap="viridis"
            )
            axes[i][1].set_title(f"Predicted Concentration (Bin {i+1})")
            plt.colorbar(im1, ax=axes[i][1])

            # Plot difference
            diff = y_true[i].cpu().numpy() - y_pred[i].cpu().numpy()
            im2 = axes[i][2].pcolormesh(lon, lat, diff, shading="auto", cmap="RdBu_r")
            axes[i][2].set_title(f"Difference (Bin {i+1})")
            plt.colorbar(im2, ax=axes[i][2])

        plt.tight_layout()
        plt.savefig(
            results_dir / "predictions" / f"prediction_epoch_{epoch}.png", dpi=150
        )
        plt.close(fig)


# Configure the model
model = PINNModel(
    in_channels=in_channels,
    out_channels=out_channels,
    modes1=12,
    modes2=12,
    width=32,
    num_layers=2,
    learning_rate=1e-3,
    weight_decay=1e-5,
    physics_weight=0.1,
    boundary_weight=0.1,
    conservation_weight=0.1,
    include_coordinates=True,
)

# Configure logger and checkpoint callback
logger = TensorBoardLogger(save_dir=str(PATH.LOGS), name="pinn_training")
checkpoint_callback = ModelCheckpoint(
    dirpath=str(PATH.CHECKPOINTS / "pinn_training"),
    filename="pinn-{epoch:03d}-{val_total_loss:.4f}",
    save_top_k=3,
    monitor="val_total_loss",
    mode="min",
)

# Configure trainer with custom callback
metrics_callback = MetricsCallback()
trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="auto",  # Use GPU if available
    devices=1,
    logger=logger,
    callbacks=[checkpoint_callback, metrics_callback],
    log_every_n_steps=10,
    check_val_every_n_epoch=10,  # Validate every 10 epochs
)

# Train the model
print(f"Starting training for {MAX_EPOCHS} epochs...")
trainer.fit(model, data_module)

# Evaluate on validation set
print("Evaluating on validation set...")
val_results = trainer.validate(model, data_module)
print(f"Final validation results: {val_results}")

# Plot training and validation losses
plt.figure(figsize=(12, 8))
plt.plot(epochs, train_losses, "b-", label="Training Loss")
plt.plot(
    val_epochs, val_losses, "r-", label="Validation Loss"
)  # Fixed: use val_epochs instead
plt.plot(epochs, physics_losses, "g-", label="Physics Loss")
plt.plot(epochs, data_losses, "y-", label="Data Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Losses")
plt.legend()
plt.grid(True)
plt.savefig(results_dir / "plots" / "losses.png", dpi=150)
plt.close()

# Plot gradients for selected parameters
plt.figure(figsize=(15, 10))
num_params_to_plot = min(10, len(param_names))  # Plot up to 10 parameters
for i, name in enumerate(param_names[:num_params_to_plot]):
    if param_grads[name]:
        plt.subplot(5, 2, i + 1)
        plt.plot(epochs, param_grads[name])
        plt.title(f"Gradient Norm: {name}")
        plt.xlabel("Epoch")
        plt.ylabel("Gradient Norm")
        plt.grid(True)
plt.tight_layout()
plt.savefig(results_dir / "gradients" / "parameter_gradients.png", dpi=150)
plt.close()

# Also create a plot for the mass conservation loss which might be very large
plt.figure(figsize=(10, 6))
plt.semilogy(
    val_epochs, [result["val_mass_conservation_loss"] for result in val_results], "r-"
)
plt.xlabel("Epoch")
plt.ylabel("Mass Conservation Loss (log scale)")
plt.title("Mass Conservation Loss")
plt.grid(True)
plt.savefig(results_dir / "plots" / "mass_conservation_loss.png", dpi=150)
plt.close()

# Save all metrics for future analysis
np.savez(
    results_dir / "training_metrics.npz",
    epochs=np.array(epochs),
    val_epochs=np.array(val_epochs),
    train_losses=np.array(train_losses),
    val_losses=np.array(val_losses),
    physics_losses=np.array(physics_losses),
    data_losses=np.array(data_losses),
    param_names=np.array(param_names),
    **{
        f"grad_{name.replace('.', '_')}": np.array(param_grads[name])
        for name in param_names
    },
)

print("Training completed!")
