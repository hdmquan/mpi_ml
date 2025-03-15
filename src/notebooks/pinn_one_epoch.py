import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data.module import MicroplasticDataModule
from src.models.fno.pinn import PINNModel
from src.utils import set_seed, PATH

set_seed()

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
logger = TensorBoardLogger(save_dir=str(PATH.LOGS), name="pinn_test")
checkpoint_callback = ModelCheckpoint(
    dirpath=str(PATH.CHECKPOINTS / "pinn_test"),
    filename="pinn-{epoch:02d}",
    save_top_k=1,
    monitor="val_total_loss",
    mode="min",
)

# Configure trainer
trainer = pl.Trainer(
    max_epochs=1,
    accelerator="auto",  # Use GPU if available
    devices=1,
    logger=logger,
    callbacks=[checkpoint_callback],
    log_every_n_steps=1,
)

# Train for one epoch
print("Starting training for one epoch...")
trainer.fit(model, data_module)

# Evaluate on validation set
print("Evaluating on validation set...")
val_results = trainer.validate(model, data_module)
print(f"Validation results: {val_results}")

# Make a prediction on a sample batch
print("Making a prediction on a sample batch...")
with torch.no_grad():
    model.eval()
    x = sample_batch["inputs"].to(model.device)
    coords = sample_batch["coords"].to(model.device)
    y_true = sample_batch["targets"].to(model.device)

    y_pred = model(x, coords)

    # Calculate MSE
    mse = torch.nn.functional.mse_loss(y_pred, y_true).item()
    print(f"Sample batch MSE: {mse:.6f}")

# Save sample predictions for visualization
sample_data = {
    "inputs": x.cpu().numpy(),
    "targets": y_true.cpu().numpy(),
    "predictions": y_pred.cpu().numpy(),
    "coords": coords.cpu().numpy(),
}

import numpy as np

np.save(str(PATH.RESULTS / "pinn_sample_predictions.npy"), sample_data)

print("One epoch test completed!")
