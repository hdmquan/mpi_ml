# %% Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.cnn import CNNPINNStream
from src.data.module import MPIDataModule
from src.utils import PATH

# %% Load data
datamodule = MPIDataModule(include_coordinates=False, batch_size=1, num_workers=4)
datamodule.setup()

# %% Load and process first model
model1 = CNNPINNStream.load_from_checkpoint(
    PATH.CHECKPOINTS / "best-model-32_64_32-128_transport.ckpt",
    map_location=torch.device("cpu"),
)
model1.eval()

all_rmse1 = []
all_r2_1 = []

for instance_idx in range(6):
    # Get a single instance
    batch = next(iter(datamodule.test_dataloader()))
    x, y, metadata = batch
    x = x.cpu()
    y = y.cpu()

    # Get prediction
    with torch.no_grad():
        y_pred = model1(x)

    # Calculate metrics
    mse = ((y.numpy() - y_pred.numpy()) ** 2).mean(axis=(0, 2, 3))
    rmse = np.sqrt(mse)

    y_true_mean = y.numpy().mean(axis=(0, 2, 3), keepdims=True)
    total_ss = ((y.numpy() - y_true_mean) ** 2).mean(axis=(0, 2, 3))
    r2 = 1 - (mse / total_ss)

    all_rmse1.append(rmse)
    all_r2_1.append(r2)

    # Clear memory
    del x, y, y_pred
    torch.cuda.empty_cache()

rmse_model1 = np.array(all_rmse1)
r2_model1 = np.array(all_r2_1)
del model1
torch.cuda.empty_cache()

# %% Load and process second model
model2 = CNNPINNStream.load_from_checkpoint(
    PATH.CHECKPOINTS / "best-model-32_64_32-128_transport.ckpt",
    map_location=torch.device("cpu"),
)
model2.eval()

all_rmse2 = []
all_r2_2 = []

for instance_idx in range(6):
    # Get a single instance
    batch = next(iter(datamodule.test_dataloader()))
    x, y, metadata = batch
    x = x.cpu()
    y = y.cpu()

    # Get prediction
    with torch.no_grad():
        y_pred = model2(x)

    # Calculate metrics
    mse = ((y.numpy() - y_pred.numpy()) ** 2).mean(axis=(0, 2, 3))
    rmse = np.sqrt(mse)

    y_true_mean = y.numpy().mean(axis=(0, 2, 3), keepdims=True)
    total_ss = ((y.numpy() - y_true_mean) ** 2).mean(axis=(0, 2, 3))
    r2 = 1 - (mse / total_ss)

    all_rmse2.append(rmse)
    all_r2_2.append(r2)

    # Clear memory
    del x, y, y_pred
    torch.cuda.empty_cache()

rmse_model2 = np.array(all_rmse2)
r2_model2 = np.array(all_r2_2)
del model2
torch.cuda.empty_cache()

# %% Plot RMSE
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i in range(6):
    ax = axes[i]
    ax.plot(rmse_model1[i], label="Model 1", color="blue")
    ax.plot(rmse_model2[i], label="Model 2", color="red")
    ax.set_title(f"Instance {i+1}")
    ax.set_xlabel("Altitude Level")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

# %% Plot R²
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i in range(6):
    ax = axes[i]
    ax.plot(r2_model1[i], label="Model 1", color="blue")
    ax.plot(r2_model2[i], label="Model 2", color="red")
    ax.set_title(f"Instance {i+1}")
    ax.set_xlabel("Altitude Level")
    ax.set_ylabel("R²")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

# %% Print average metrics
print("Model 1 Average Metrics:")
print(f"RMSE: {rmse_model1.mean():.6f}")
print(f"R²: {r2_model1.mean():.6f}")

print("\nModel 2 Average Metrics:")
print(f"RMSE: {rmse_model2.mean():.6f}")
print(f"R²: {r2_model2.mean():.6f}")
