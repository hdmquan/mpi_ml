# %% Imports
import pytorch_lightning as pl
from src.models.fno.pinn import PINNModel
from src.data.module import MPIDataModule
from src.utils.plotting import plot_layer, plot_long_cut

# %% Data
datamodule = MPIDataModule(batch_size=1, num_workers=0)
datamodule.setup()

# %% Model
model = PINNModel()

# %% Training
trainer = pl.Trainer(
    max_epochs=5,
    devices=1,
    precision=16,
    enable_checkpointing=True,
    enable_progress_bar=True,
    enable_model_summary=True,
)

trainer.fit(model, datamodule)

# %% Testing
test_results = trainer.test(model, datamodule)
print(f"Test Results: {test_results}")

# %% Get sample batch and predict
batch = next(iter(datamodule.test_dataloader()))
x, (mmr, _, _), cell_area = batch  # Unpack the targets tuple correctly
y_pred = model(x)

# %% Visualize level 1
lev = 1

# Ground truth
plot_layer(mmr, lev - 1)

# Predictions
plot_layer(y_pred, lev - 1)

# %% Visualize longitude cuts
# Ground truth
plot_long_cut(mmr, mmr.shape[-1] // 2)

# Predictions
plot_long_cut(y_pred, y_pred.shape[-1] // 2)
