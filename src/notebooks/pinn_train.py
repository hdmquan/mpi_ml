# %% Imports
import torch
from tqdm import tqdm
from src.models.fno.pinn import PINNModel
from src.data.module import MPIDataModule
from src.utils.plotting import plot_layer, plot_long_cut

# %% Data
datamodule = MPIDataModule(batch_size=1, num_workers=4)
datamodule.setup()

print("Setup complete")

train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()

# %% Model and optimization setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PINNModel().to(device)

optimizer = model.configure_optimizers()["optimizer"]
scheduler = model.configure_optimizers()["lr_scheduler"]

# %% Training configuration
num_epochs = 1
best_val_loss = float("inf")

# %% Training loop
print(f"Training for {num_epochs} epochs")

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_losses = []

    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]") as pbar:
        for batch in pbar:
            print(".", end="")
            x, y, cell_area = [b.to(device) for b in batch]

            optimizer.zero_grad()
            y_pred = model(x)
            losses = model.compute_loss(x, y_pred, y, cell_area)
            total_loss = losses["total"]

            total_loss.backward()
            optimizer.step()

            train_losses.append(total_loss.item())
            pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

    print()

    # Validation phase
    model.eval()
    val_losses = []

    with torch.no_grad():
        with tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]") as pbar:
            for batch in pbar:
                x, y, cell_area = [b.to(device) for b in batch]

                y_pred = model(x)
                losses = model.compute_loss(x, y_pred, y, cell_area)
                total_loss = losses["total"]

                val_losses.append(total_loss.item())
                pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

    # Calculate average losses
    avg_train_loss = sum(train_losses) / len(train_losses)
    avg_val_loss = sum(val_losses) / len(val_losses)

    print(
        f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
    )

    # Learning rate scheduling
    scheduler.step()

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pt")

# %% Testing
model.eval()
test_losses = []
test_rmse = []

with torch.no_grad():
    with tqdm(test_loader, desc="Testing") as pbar:
        for batch in pbar:
            x, y, cell_area = [b.to(device) for b in batch]

            y_pred = model(x)
            losses = model.compute_loss(x, y_pred, y, cell_area)
            rmse = torch.sqrt(torch.nn.functional.mse_loss(y_pred, y))

            test_losses.append(losses["total"].item())
            test_rmse.append(rmse.item())
            pbar.set_postfix(
                {"loss": f'{losses["total"].item():.4f}', "rmse": f"{rmse.item():.4f}"}
            )

avg_test_loss = sum(test_losses) / len(test_losses)
avg_test_rmse = sum(test_rmse) / len(test_rmse)
print(f"Test Results: Loss = {avg_test_loss:.4f}, RMSE = {avg_test_rmse:.4f}")

# %% Visualization
batch = next(iter(test_loader))
x, y, cell_area = [b.to(device) for b in batch]
y_pred = model(x)

# Ground truth
plot_layer(y.cpu(), 0, save=True)  # Level 1

# Predictions
plot_layer(y_pred.cpu(), 0, save=True)  # Level 1

# Ground truth longitude cuts
plot_long_cut(y.cpu(), y.shape[-1] // 2, save=True)

# Predictions longitude cuts
plot_long_cut(y_pred.cpu(), y_pred.shape[-1] // 2, save=True)

# %%
