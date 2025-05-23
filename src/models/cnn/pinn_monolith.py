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


class CNNPINN(Base):
    def __init__(
        self,
        in_channels=5,
        out_channels=6,
        hidden_channels=[32, 64, 32],  # Lightweight channel configuration
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
        """
        Lightweight 3D CNN-based PINN model to predict MMR (Mass Mixing Ratio).

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (particle sizes)
            hidden_channels: List of channel sizes for hidden layers
            kernel_size: Size of convolutional kernels
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            mmr_weight: Weight for MMR loss term
            conservation_weight: Weight for conservation loss term
            physics_weight: Weight for physics loss term
            settling_velocities: Optional custom settling velocities
            output_altitude_dim: Optional output altitude dimension (if different from input)
        """
        super().__init__(
            mmr_weight=mmr_weight,
            conservation_weight=conservation_weight,
            physics_weight=physics_weight,
            settling_velocities=settling_velocities,
            **kwargs,
        )

        self.save_hyperparameters()

        # Build CNN layers
        layers = []

        # Input layer
        layers.append(ConvBlock(in_channels, hidden_channels[0], kernel_size))

        # Hidden layers
        for i in range(len(hidden_channels) - 1):
            layers.append(
                ConvBlock(hidden_channels[i], hidden_channels[i + 1], kernel_size)
            )

        # Output layer (no activation/batch norm)
        layers.append(
            nn.Conv3d(
                hidden_channels[-1], out_channels, kernel_size=kernel_size, padding=1
            )
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x, coords=None):
        """
        Forward pass returning MMR predictions.

        Args:
            x: Input tensor [batch_size, channels, altitude, latitude, longitude]
            coords: Not used in CNN implementation

        Returns:
            tensor: MMR predictions [batch_size, n_particles, altitude, latitude, longitude]
                   with values constrained to [0,1] via sigmoid activation
        """
        output = self.network(x)

        # Apply sigmoid to constrain values between 0 and 1
        output = torch.sigmoid(output)

        # Handle different altitude dimensions if needed
        if (
            self.hparams.output_altitude_dim is not None
            and self.hparams.output_altitude_dim != output.shape[2]
        ):
            # Resize in the altitude dimension using interpolation
            output = F.interpolate(
                output,
                size=(
                    self.hparams.output_altitude_dim,
                    output.shape[3],
                    output.shape[4],
                ),
                mode="trilinear",
                align_corners=False,
            )

        return output


if __name__ == "__main__":
    import time
    from src.utils import count_parameters

    # Test model
    model = CNNPINN(in_channels=11, output_altitude_dim=48 + 2)  # MMR + WetDep + DryDep
    print(f"Number of parameters: {count_parameters(model)}")

    # Test forward pass
    x = torch.randn(1, 11, 48, 384, 576)
    start_time = time.time()
    predictions = model(x)
    print(f"Forward pass time: {time.time() - start_time:.3f} seconds")
    print(f"Output shape: {predictions.shape}")

    # Test loss computation
    y = torch.randn(1, 6, 50, 384, 576)
    losses = model.compute_loss(x, predictions, y)
    print("Loss computation successful")

    start_time = time.time()
    losses["total"].backward()
    print(f"Backward pass time: {time.time() - start_time:.3f} seconds")
