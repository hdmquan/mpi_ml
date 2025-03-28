import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from loguru import logger

from src.models.fno.core import FNOModel3D
from src.utils import set_seed

set_seed()


class PINNModel(pl.LightningModule):

    def __init__(
        self,
        # NOTE: Troposphere is not used since it's zeros
        # Order: PS, (wind)U, (wind)V, T, Q, emissions
        # Translate: Pressure, Wind East, Wind North, Temperature, Humidity, Source term
        in_channels=7,
        out_channels=6,  # 6 sizes
        # FNO settings
        modes1=8,  # altitude modes
        modes2=12,  # latitude modes
        modes3=12,  # longitude modes
        width=32,
        num_layers=2,
        learning_rate=1e-3,
        weight_decay=1e-5,
        # PINN loss weights
        mmr_weight=1.0,
        conservation_weight=0.1,
        physics_weight=0.1,
        #
        settling_velocities=None,
        include_coordinates=True,
    ):
        """
        PINN model to predict MMR (Mass Mixing Ratio) with altitude.
        """
        super(PINNModel, self).__init__()

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
            include_coordinates=include_coordinates,
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

    def forward(self, x, coords=None):
        """
        Forward pass returning MMR predictions.

        Args:
            x: Input tensor [batch_size, channels, altitude, latitude, longitude]
            coords: Optional coordinate grid

        Returns:
            tensor: MMR predictions [batch_size, n_particles, altitude, latitude, longitude]
        """
        if coords is None:
            coords = self.fno_model.get_grid(
                (x.size(-3), x.size(-2), x.size(-1)), device=x.device
            ).expand(x.size(0), -1, -1, -1, -1)

        return self.fno_model(x, coords)

    def compute_loss(self, x, mmr_pred, mmr_true, cell_area, dt=1.0):
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

        # L_mass = (∫∫∫ ρ·MMR dV - ∫∫ Emissions dA)²
        # Store the original emissions shape before summation
        # ρ is ignore since this is a loss
        mass_true = (mmr_true * cell_area[None, None, None, :, :]).sum(dim=(-2, -1))
        mass_pred = (mmr_pred * cell_area[None, None, None, :, :]).sum(dim=(-2, -1))
        emissions_sum = (emissions * cell_area[None, None, None, :, :]).sum(
            dim=(-2, -1)
        )

        mass_conservation_error = (mass_pred - mass_true) - emissions_sum
        total_mass = torch.abs(mass_true).mean() + torch.abs(emissions_sum).mean()
        losses["mass_conservation"] = torch.mean(
            (mass_conservation_error / (total_mass + 1e-8)) ** 2
        )

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

        logger.debug(f"Laplacian: {laplacian_C.shape}")

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
            + self.hparams.conservation_weight * losses["mass_conservation"]
            + self.hparams.physics_weight * losses["transport"]
        )

        losses["total"] = total_loss
        return losses

    def training_step(self, batch, batch_idx) -> torch.Tensor:

        x, targets, cell_area = batch

        logger.debug(f"X: {x.shape}")
        logger.debug(f"Targets: {targets.shape}")
        logger.debug(f"Cell area: {cell_area.shape}")

        predictions = self(x)
        losses = self.compute_loss(x, predictions, targets, cell_area)

        for name, value in losses.items():
            self.log(f"train_{name}", value, prog_bar=name == "total")

        return losses["total"]

    def validation_step(self, batch, batch_idx) -> torch.Tensor:

        x, y_true, cell_area = batch

        y_pred = self(x)
        losses = self.compute_loss(x, y_pred, y_true, cell_area)

        for name, value in losses.items():
            self.log(f"val_{name}", value, prog_bar=name == "total")

        return losses["total"]

    def test_step(self, batch, batch_idx) -> torch.Tensor:

        x, y_true, cell_area = batch

        y_pred = self(x)
        losses = self.compute_loss(x, y_pred, y_true, cell_area)

        rmse = torch.sqrt(F.mse_loss(y_pred, y_true))

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
            T_max=100,  # TODO: Adjust
            eta_min=self.hparams.learning_rate / 10,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_total_loss",
        }


if __name__ == "__main__":
    import time

    model = PINNModel()

    # batch_size, features, altitude, latitude, longitude
    x = torch.randn(1, 7, 48, 600, 400, requires_grad=True)

    start_time = time.time()
    predictions = model(x)
    end_time = time.time()
    print(f"Time taken for forward pass: {end_time - start_time} seconds")

    print(f"Predictions: {predictions.shape}")

    # batch_size, features, altitude, latitude, longitude
    sample_y = torch.randn(1, 6, 48, 600, 400)

    start_time = time.time()
    # Sample loss
    losses = model.compute_loss(
        x,
        predictions,
        sample_y,
        torch.randn(600, 400),
    )
    end_time = time.time()
    print(f"Time taken for loss computation: {end_time - start_time} seconds")

    start_time = time.time()
    losses["total"].backward()
    end_time = time.time()
    print(f"Time taken for backward pass: {end_time - start_time} seconds")

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.isnan(
            param.grad
        ).any(), f"Parameter {name} has NaN gradient values"
        assert not torch.isinf(
            param.grad
        ).any(), f"Parameter {name} has Inf gradient values"

    logger.success("Gradient check passed successfully!")
