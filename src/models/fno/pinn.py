import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from loguru import logger

from src.models.fno.core import FNOModel3D
from src.models.base import Base
from src.utils import set_seed

set_seed()

# Original settings
# modes1=8,  # altitude modes
# modes2=12,  # latitude modes
# modes3=12,  # longitude modes
# width=32,


class FNOPINN(Base):

    def __init__(
        self,
        # NOTE: Troposphere is not used since it's zeros
        # Order: PS, (wind)U, (wind)V, T, Q, emissions
        # Translate: Pressure, Wind East, Wind North, Temperature, Humidity, Source term
        in_channels=5,
        out_channels=6,  # 6 sizes
        # FNO settings
        modes1=4,  # altitude modes
        modes2=12,  # latitude modes
        modes3=12,  # longitude modes
        width=8,
        num_layers=3,
        learning_rate=1e-3,
        weight_decay=1e-5,
        # PINN loss weights
        mmr_weight=1.0,
        conservation_weight=0.1,
        physics_weight=0.1,
        #
        settling_velocities=None,
        **kwargs,
    ):
        """
        FNO-based PINN model to predict MMR (Mass Mixing Ratio) with altitude.
        """
        # Call parent constructor with PINN-specific parameters
        super().__init__(
            mmr_weight=mmr_weight,
            conservation_weight=conservation_weight,
            physics_weight=physics_weight,
            settling_velocities=settling_velocities,
            **kwargs,
        )

        self.save_hyperparameters()

        self.n_particles = 6

        self.fno_model = FNOModel3D(
            in_channels=in_channels,
            out_channels=out_channels,
            modes1=modes1,
            modes2=modes2,
            modes3=modes3,
            width=width,
            num_layers=num_layers,
        )

        if settling_velocities is None:
            # Default settling velocities for different plastic size bins (m/s)
            # Derived from the supplementary material of the paper
            self.settling_velocities = torch.tensor(
                [0.00097, 0.0087, 0.097, 0.39, 2.7, 4.98], dtype=torch.float32
            )
        else:
            self.settling_velocities = torch.tensor(
                settling_velocities, dtype=torch.float32
            )

        # Buffer the settling velocity
        self.register_buffer("settling_vel", self.settling_velocities)

    def forward(self, x):
        """
        Forward pass returning MMR predictions.

        Args:
            x: Input tensor [batch_size, channels, altitude, latitude, longitude]

        Returns:
            tensor: MMR predictions [batch_size, n_particles, altitude, latitude, longitude]
        """

        return self.fno_model(x)


if __name__ == "__main__":
    import time

    from src.utils import count_parameters
    from src.utils.memory import print_memory_allocated

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_memory_allocated()

    model = FNOPINN().to(device)

    print_memory_allocated()

    print(f"Number of parameters: {count_parameters(model)}")

    # batch_size, features, altitude, latitude, longitude
    x = torch.randn(1, 5, 48, 600, 400, requires_grad=True).to(device)

    print_memory_allocated()

    start_time = time.time()
    predictions = model(x)
    end_time = time.time()
    print(f"Time taken for forward pass: {end_time - start_time} seconds")

    print(f"Predictions: {predictions.shape}")

    # batch_size, features, altitude, latitude, longitude
    sample_y = torch.randn(1, 6, 48, 600, 400).to(device)

    start_time = time.time()
    # Sample loss
    losses = model.compute_loss(
        x,
        predictions,
        sample_y,
    )
    end_time = time.time()
    print(f"Time taken for loss computation: {end_time - start_time} seconds")
    print_memory_allocated()

    start_time = time.time()
    losses["total"].backward()
    end_time = time.time()
    print(f"Time taken for backward pass: {end_time - start_time} seconds")
    print_memory_allocated()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.isnan(
            param.grad
        ).any(), f"Parameter {name} has NaN gradient values"
        assert not torch.isinf(
            param.grad
        ).any(), f"Parameter {name} has Inf gradient values"

    logger.success("Gradient check passed successfully!")
