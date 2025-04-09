# %% Imports
import torch
import torch.nn as nn
from tqdm import tqdm
from src.data.module import MPIDataModule


# %% Simple test model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(7, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 6, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.conv_layers(x)

    def compute_loss(self, x, y_pred, y, cell_area):
        mse_loss = torch.nn.functional.mse_loss(y_pred, y)
        return {"total": mse_loss}


# %% Data
datamodule = MPIDataModule(batch_size=1, num_workers=4)
datamodule.setup()
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()

# %% Model and optimization setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# %% Training configuration
num_epochs = 2
print(f"Training for {num_epochs} epochs")

# %% Training loop
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_losses = []

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
        x, y, cell_area = [b.to(device) for b in batch]

        optimizer.zero_grad()
        y_pred = model(x)
        loss = model.compute_loss(x, y_pred, y, cell_area)["total"]

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    avg_train_loss = sum(train_losses) / len(train_losses)
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

# %% Test prediction
model.eval()
with torch.no_grad():
    batch = next(iter(val_loader))
    x, y, cell_area = [b.to(device) for b in batch]
    y_pred = model(x)
    loss = model.compute_loss(x, y_pred, y, cell_area)["total"]
    print(f"Test prediction loss: {loss.item():.4f}")


# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"Number of parameters: {count_parameters(model)}")

# %%
