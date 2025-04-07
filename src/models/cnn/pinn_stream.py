import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import Base
from src.utils import set_seed

set_seed()


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CNNPINNStream(Base):
    def __init__(
        self,
        in_channels=11,
        mmr_out_channels=6,
        dep_out_channels=2,
        hidden_channels=[8, 16, 8],
        kernel_size=3,
        learning_rate=1e-3,
        weight_decay=1e-5,
        mmr_weight=1.0,
        conservation_weight=0.1,
        physics_weight=0.1,
        settling_velocities=None,
        output_altitude_dim=None,
        **kwargs,
    ):
        super().__init__(
            mmr_weight=mmr_weight,
            conservation_weight=conservation_weight,
            physics_weight=physics_weight,
            settling_velocities=settling_velocities,
            **kwargs,
        )

        self.save_hyperparameters()

        # Shared 3D CNN Encoder
        layers = []
        layers.append(ConvBlock(in_channels, hidden_channels[0], kernel_size))
        for i in range(len(hidden_channels) - 1):
            layers.append(
                ConvBlock(hidden_channels[i], hidden_channels[i + 1], kernel_size)
            )
        self.encoder = nn.Sequential(*layers)

        # MMR Head (3D CNN)
        self.mmr_head = nn.Conv3d(
            hidden_channels[-1], mmr_out_channels, kernel_size=kernel_size, padding=1
        )

        # Deposition Head (2D CNNs)
        self.dep_conv1 = nn.Conv2d(
            hidden_channels[-1] + mmr_out_channels, 16, kernel_size=3, padding=1
        )
        self.dep_conv2 = nn.Conv2d(16, dep_out_channels, kernel_size=3, padding=1)

    def forward(self, x, coords=None):
        features = self.encoder(x)

        # MMR prediction
        mmr = torch.sigmoid(self.mmr_head(features))

        if (
            self.hparams.output_altitude_dim is not None
            and self.hparams.output_altitude_dim != mmr.shape[2]
        ):
            mmr = F.interpolate(
                mmr,
                size=(self.hparams.output_altitude_dim, mmr.shape[3], mmr.shape[4]),
                mode="trilinear",
                align_corners=False,
            )

        # Collapse 3D features to 2D (e.g., mean over altitude)
        pooled = features.mean(dim=2)  # [B, C, H, W]
        mmr_surface = mmr.sum(dim=2)  # [B, 6, H, W]

        dep_input = torch.cat([pooled, mmr_surface], dim=1)  # [B, C+6, H, W]
        dep_hidden = F.relu(self.dep_conv1(dep_input))
        deposition = torch.sigmoid(self.dep_conv2(dep_hidden))  # [B, 2, H, W]

        # Convert 2D deposition to 3D by expanding altitude dimension
        # Shape will be [B, 2, 1, H, W]
        deposition_3d = deposition.unsqueeze(2)

        # Expand deposition to match mmr's altitude dimension
        # Shape will be [B, 2, altitude, H, W]
        if self.hparams.output_altitude_dim is not None:
            alt_dim = self.hparams.output_altitude_dim
        else:
            alt_dim = mmr.shape[2]

        deposition_expanded = deposition_3d.expand(-1, -1, alt_dim, -1, -1)

        # For altitude layers beyond the surface layer, set deposition values to zero
        # Only the bottom-most layer will have non-zero values
        bottom_mask = torch.zeros_like(deposition_expanded)
        bottom_mask[:, :, 0, :, :] = 1.0  # Only the first altitude layer has deposition
        deposition_expanded = deposition_expanded * bottom_mask

        # Concatenate along the channel dimension
        # Shape will be [B, mmr_out_channels + dep_out_channels, altitude, H, W]
        predictions = torch.cat([mmr, deposition_expanded], dim=1)

        return predictions


if __name__ == "__main__":
    import time
    from src.utils import count_parameters

    model = CNNPINNStream(output_altitude_dim=50, use_physics_loss=True)
    print(f"Number of parameters: {count_parameters(model)}")

    x = torch.randn(1, 11, 48, 384, 576)
    start_time = time.time()
    predictions = model(x)
    print(f"Forward pass time: {time.time() - start_time:.3f} seconds")
    print(f"Predictions shape: {predictions.shape}")

    # The combined output tensor shape should be [B, mmr_out_channels + dep_out_channels, altitude, H, W]
    # In this case: [1, 8, 50, 384, 576] = [1, 6+2, 50, 384, 576]

    # Create a mock ground truth tensor for testing the loss
    # It needs to match the output shape of the model
    y = torch.randn(1, 8, 50, 384, 576)  # 6 MMR channels + 2 deposition channels

    # Test loss computation
    predictions = model(x)
    losses = model.compute_loss(x, predictions, y)
    print("Loss computation successful")

    start_time = time.time()
    losses["total"].backward()
    print(f"Backward pass time: {time.time() - start_time:.3f} seconds")
