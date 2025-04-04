import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from abc import ABC, abstractmethod
import torch.nn as nn
from loguru import logger

from src.utils import set_seed

set_seed()


class Base(pl.LightningModule, ABC):

    def __init__(
        self,
        in_channels=7,
        out_channels=6,
        mmr_weight=1.0,
        physics_weight=1.0,
        settling_velocities=None,
        include_coordinates=True,
        use_physics_loss=False,
        **kwargs,
    ):
        """
        Abstract base model class.
        """
        super(Base, self).__init__()
        self.save_hyperparameters()

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
        self.register_buffer("settling_vel", self.settling_velocities)

    @abstractmethod
    def forward(self, x, coords=None):
        """
        Forward pass returning MMR predictions.

        Args:
            x: Input tensor [batch_size, channels, altitude, latitude, longitude]
            coords: Optional coordinate grid

        Returns:
            tensor: MMR predictions [batch_size, n_particles, altitude, latitude, longitude]
        """
        pass

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, targets = batch
        predictions = self(x)
        losses = self.compute_loss(x, predictions, targets)

        # Log all loss components
        for name, value in losses.items():
            self.log(f"train_{name}", value, prog_bar=name == "total")

        return losses["total"]

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x, y_true = batch
        y_pred = self(x)
        losses = self.compute_loss(x, y_pred, y_true)

        # Log all loss components
        for name, value in losses.items():
            self.log(f"val_{name}", value, prog_bar=name == "total")

        return losses["total"]

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        x, y_true = batch
        y_pred = self(x)
        losses = self.compute_loss(x, y_pred, y_true)
        rmse = torch.sqrt(F.mse_loss(y_pred, y_true))

        # Log all loss components
        for name, value in losses.items():
            self.log(f"test_{name}", value)
        self.log("test_rmse", rmse)

        return losses["total"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=self.hparams.learning_rate / 10,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_total",
        }

    def compute_loss(self, x, mmr_pred, mmr_true, dt=1.0):
        if self.hparams.use_physics_loss:
            return self.compute_physics_loss(x, mmr_pred, mmr_true, dt)
        else:
            # Shape [1, 6, 50, 384, 576]
            # MMR [1, 6, :-2, 384, 576]
            # Dry + Wet Deposition [1, -2:, 50, 384, 576]

            # Split the prediction and ground truth into MMR and depositions
            n_particles = mmr_pred.shape[1]
            mmr_particles = mmr_pred[:, : n_particles - 2]  # All except last 2 channels
            deposition_particles = mmr_pred[
                :, -2:
            ]  # Last 2 channels (dry and wet deposition)

            mmr_true_particles = mmr_true[:, : n_particles - 2]
            deposition_true_particles = mmr_true[:, -2:]

            # Create altitude weights (exponentially decreasing with height) for MMR
            altitude_dim = mmr_pred.shape[2]
            altitude_weights = torch.exp(
                -torch.arange(altitude_dim, device=mmr_pred.device) * 0.5
            )
            # Normalize weights to range from 2 to 1
            altitude_weights = 1 + (altitude_weights - altitude_weights.min()) / (
                altitude_weights.max() - altitude_weights.min()
            )
            altitude_weights = altitude_weights.view(1, 1, -1, 1, 1)

            # Apply weights to squared differences for MMR
            mmr_squared_diff = (mmr_particles - mmr_true_particles) ** 2
            mmr_weighted_squared_diff = mmr_squared_diff * altitude_weights
            mmr_loss = mmr_weighted_squared_diff.mean()

            # Apply weight of 2 to depositions
            deposition_loss = 2.0 * F.mse_loss(
                deposition_particles, deposition_true_particles
            )

            # Combine losses
            total_loss = mmr_loss + deposition_loss

            return {"total": total_loss, "mmr": mmr_loss, "deposition": deposition_loss}

    def compute_physics_loss(self, x, mmr_pred, mmr_true, dt=1.0):
        """
        Compute physics-informed losses for MMR predictions.

        Args:
            x: Input features [batch_size, in_channels, altitude, latitude, longitude]
            mmr_pred: MMR predictions [batch_size, n_particles, altitude, latitude, longitude]
            mmr_true: MMR true values [batch_size, n_particles, altitude, latitude, longitude]
            cell_area: Area of each grid cell [height, width]
            dt: Time step (right now we don't do time-dependent simulations)
        """
        batch_size, n_particles, altitude, latitude, longitude = mmr_pred.shape
        device = mmr_pred.device

        ps = x[:, [0]]  # Pressure
        u = x[:, [1]]  # Wind East
        v = x[:, [2]]  # Wind North
        emissions = x[:, [-1]]

        # logger.info("Initial shapes")

        # logger.debug(f"Pressure: {ps.shape}")
        # logger.debug(f"Wind East: {u.shape}")
        # logger.debug(f"Wind North: {v.shape}")
        # logger.debug(f"Emissions: {emissions.shape}")

        # Settling velocity
        v_s = (
            self.settling_vel.view(1, -1, 1, 1, 1)
            .expand(batch_size, -1, altitude, latitude, longitude)
            .to(device)
        )

        # logger.debug(f"Settling velocity: {v_s.shape}")

        losses = {}

        # L_mmr = (1/N) Σ (MMR pred - MMR true)^2
        losses["mmr"] = F.mse_loss(mmr_pred, mmr_true)

        # NOTE: Turn out it varied over months.
        # L_mass = (∫∫∫ ρ·MMR dV - ∫∫ Emissions dA)²
        # Store the original emissions shape before summation
        # ρ is ignore since this is a loss
        # mass_true = (mmr_true * cell_area[None, None, None, :, :]).sum(dim=(-2, -1))
        # mass_pred = (mmr_pred * cell_area[None, None, None, :, :]).sum(dim=(-2, -1))
        # emissions_sum = (emissions * cell_area[None, None, None, :, :]).sum(
        #     dim=(-2, -1)
        # )

        # mass_conservation_error = (mass_pred - mass_true) - emissions_sum
        # total_mass = torch.abs(mass_true).mean() + torch.abs(emissions_sum).mean()
        # losses["mass_conservation"] = torch.mean(
        #     (mass_conservation_error / (total_mass + 1e-8)) ** 2
        # )

        # logger.info("Transport loss")

        # L_transport = ∫∫∫ (∂C/∂t + u·∂C/∂x + v·∂C/∂y + w·∂C/∂z - K_h∇²C - K_v∂²C/∂z² - S)² dV dt
        dC_dt = (mmr_pred - mmr_true) / dt

        # Spatial derivatives (central differences with periodic padding)
        mmr_pad = F.pad(mmr_pred, (1, 1, 1, 1, 0, 0), mode="circular")
        dx, dy, dz = 1.0, 1.0, 1.0

        dC_dx = (mmr_pad[:, :, :, 1:-1, 2:] - mmr_pad[:, :, :, 1:-1, :-2]) / (2 * dx)
        dC_dy = (mmr_pad[:, :, :, 2:, 1:-1] - mmr_pad[:, :, :, :-2, 1:-1]) / (2 * dy)

        # Vertical derivatives (zero-padding for top/bottom boundaries)
        mmr_pad_z = F.pad(mmr_pred, (0, 0, 0, 0, 1, 1), mode="replicate")
        dC_dz = (mmr_pad_z[:, :, 2:, :, :] - mmr_pad_z[:, :, :-2, :, :]) / (2 * dz)

        # logger.debug(f"MMR pad: {mmr_pad.shape}")
        # logger.debug(f"dC_dt: {dC_dt.shape}")
        # logger.debug(f"dC_dx: {dC_dx.shape}")
        # logger.debug(f"dC_dy: {dC_dy.shape}")

        # Horizontal diffusion term (Laplace)
        d2C_dx2 = (
            mmr_pad[:, :, :, 1:-1, 2:] - 2 * mmr_pred + mmr_pad[:, :, :, 1:-1, :-2]
        ) / (dx**2)
        d2C_dy2 = (
            mmr_pad[:, :, :, 2:, 1:-1] - 2 * mmr_pred + mmr_pad[:, :, :, :-2, 1:-1]
        ) / (dy**2)
        d2C_dz2 = (
            mmr_pad_z[:, :, 2:, :, :] - 2 * mmr_pred + mmr_pad_z[:, :, :-2, :, :]
        ) / (dz**2)

        laplacian_C = d2C_dx2 + d2C_dy2 + d2C_dz2

        # logger.debug(f"Laplacian: {laplacian_C.shape}")

        Kh = 1.0  # Horizontal diffusion coefficient
        Kv = 0.1  # Vertical diffusion coefficient
        diffusion_term = Kh * (d2C_dx2 + d2C_dy2) + Kv * d2C_dz2

        # Expand emissions to match particle and altitude dimensions
        emissions_grid = emissions.expand(-1, mmr_pred.size(1), altitude, -1, -1)

        transport_residual = (
            dC_dt
            + u * dC_dx
            + v * dC_dy
            + v_s * dC_dz
            - diffusion_term
            - emissions_grid
        )
        losses["transport"] = torch.mean(transport_residual**2)

        # Total loss
        total_loss = (
            self.hparams.mmr_weight * losses["mmr"]
            # + self.hparams.conservation_weight * losses["mass_conservation"]
            + self.hparams.physics_weight * losses["transport"]
        )

        losses["total"] = total_loss
        return losses


class LinearModel(Base):
    def __init__(
        self,
        in_channels=7,
        out_channels=6,
        mmr_weight=1.0,
        conservation_weight=0.1,
        physics_weight=0.1,
        use_physics_loss=False,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            mmr_weight=mmr_weight,
            conservation_weight=conservation_weight,
            physics_weight=physics_weight,
            use_physics_loss=use_physics_loss,
            **kwargs,
        )

        # Simple linear layer that preserves spatial dimensions
        self.linear = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, coords=None):
        # x shape: [batch_size, in_channels, altitude, latitude, longitude]
        return self.linear(x)


if __name__ == "__main__":
    from src.data.module import MPIDataModule

    # Initialize data
    datamodule = MPIDataModule(batch_size=1)
    datamodule.setup()

    # Get one batch of data
    train_loader = datamodule.train_dataloader()
    x, mmr_true = next(iter(train_loader))

    # Create model instances - one with physics loss and one without
    model_no_physics = LinearModel(
        in_channels=5,
        out_channels=6,
        mmr_weight=1.0,
        conservation_weight=0.1,
        physics_weight=0.1,
        use_physics_loss=False,
    )

    model_with_physics = LinearModel(
        in_channels=5,
        out_channels=6,
        mmr_weight=1.0,
        conservation_weight=0.1,
        physics_weight=0.1,
        use_physics_loss=True,
    )

    # Test forward pass
    mmr_pred_no_physics = model_no_physics(x)
    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Output shape: {mmr_pred_no_physics.shape}")
    logger.info(f"Target shape: {mmr_true.shape}")

    # Compute losses for both models
    losses_no_physics = model_no_physics.compute_loss(x, mmr_pred_no_physics, mmr_true)
    logger.info("\nLosses without physics:")
    for name, value in losses_no_physics.items():
        logger.info(f"{name}: {value.item():.6f}")

    mmr_pred_physics = model_with_physics(x)
    losses_physics = model_with_physics.compute_physics_loss(
        x, mmr_pred_physics, mmr_true
    )
    logger.info("\nLosses with physics:")
    for name, value in losses_physics.items():
        logger.info(f"{name}: {value.item():.6f}")

    # Analyze the predictions
    logger.info("\nPrediction statistics:")
    logger.info(
        f"No physics - min: {mmr_pred_no_physics.min():.6f}, max: {mmr_pred_no_physics.max():.6f}"
    )
    logger.info(
        f"With physics - min: {mmr_pred_physics.min():.6f}, max: {mmr_pred_physics.max():.6f}"
    )
    logger.info(f"True values - min: {mmr_true.min():.6f}, max: {mmr_true.max():.6f}")
