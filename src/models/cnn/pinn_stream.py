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
        hidden_channels=[64, 96, 64],
        mmr_hidden_channels=32,
        kernel_size=3,
        learning_rate=1e-3,
        weight_decay=1e-5,
        mmr_weight=1.0,
        conservation_weight=0.1,
        physics_weight=0.1,
        settling_velocities=None,
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
            hidden_channels[-1] + mmr_out_channels,
            mmr_hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.dep_conv2 = nn.Conv2d(
            mmr_hidden_channels, dep_out_channels, kernel_size=3, padding=1
        )

    def forward(self, x, coords=None):
        features = self.encoder(x)

        # MMR prediction
        mmr = torch.sigmoid(self.mmr_head(features))

        # Preserve the original altitude dimension (48)
        # We'll remove the interpolation code that changes it to output_altitude_dim (50)

        # Collapse 3D features to 2D (e.g., mean over altitude)
        pooled = features.mean(dim=2)  # [B, C, H, W]
        mmr_surface = mmr.sum(dim=2)  # [B, 6, H, W]

        dep_input = torch.cat([pooled, mmr_surface], dim=1)  # [B, C+6, H, W]
        dep_hidden = F.relu(self.dep_conv1(dep_input))
        deposition = torch.sigmoid(self.dep_conv2(dep_hidden))  # [B, 2, H, W]

        # logger.debug(f"MMR shape: {mmr.shape}")
        # logger.debug(f"Deposition shape: {deposition.shape}")

        # Get altitude dimension directly from mmr (which should be the same as input.shape[2])
        alt_dim = mmr.shape[2]  # This should be 48

        # Create output tensor with shape [B, 6, altitude+2, H, W]
        batch_size, _, _, height, width = mmr.shape
        output = torch.zeros(
            batch_size, 6, alt_dim + 2, height, width, device=mmr.device
        )

        # Copy mmr data to the altitude layers
        output[:, :, :alt_dim, :, :] = mmr

        # Add deposition data to the last two layers
        # First deposition channel goes to altitude+0
        output[:, :, alt_dim, :, :] = (
            deposition[:, 0, :, :].unsqueeze(1).expand(-1, 6, -1, -1)
        )
        # Second deposition channel goes to altitude+1
        output[:, :, alt_dim + 1, :, :] = (
            deposition[:, 1, :, :].unsqueeze(1).expand(-1, 6, -1, -1)
        )

        return output


if __name__ == "__main__":
    import time
    from loguru import logger
    from src.utils import count_parameters

    # Remove the output_altitude_dim parameter to use the original altitude dimension
    model = CNNPINNStream(use_physics_loss=True)
    logger.info(f"Number of parameters: {count_parameters(model)}")

    alt = 48
    lat = 384
    lon = 576 // 2

    x = torch.randn(1, 11, alt, lat, lon)
    start_time = time.time()

    predictions = model(x)
    logger.info(f"Forward pass time: {time.time() - start_time:.3f} seconds")
    logger.debug(f"Predictions shape: {predictions.shape}")

    # The output tensor shape should be [B, 6, altitude+2, H, W]
    # In this case: [1, 6, 50, 384, 576] = [1, 6, 48+2, 384, 576]

    # Create a mock ground truth tensor for testing the loss
    # It needs to match the output shape of the model
    y = torch.randn(1, 6, alt + 2, lat, lon)

    # Test loss computation
    losses = model.compute_loss(x, predictions, y)
    logger.info("Loss computation successful")

    start_time = time.time()
    losses["total"].backward()
    logger.info(f"Backward pass time: {time.time() - start_time:.3f} seconds")
