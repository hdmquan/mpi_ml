import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .core import FNOModel
import numpy as np
from src.utils import set_seed

set_seed()


class PINNModel(pl.LightningModule):

    def __init__(
        self,
        in_channels=7,  # Default: PS, (wind)U, (wind)V, T, Q, TROPLEV, emissions
        out_channels=6,  # 6 size
        modes1=12,  # No. Fourier modes in 1st dim
        modes2=12,  # No. Fourier modes in 2nd dim
        width=32,
        num_layers=2,
        learning_rate=1e-3,
        weight_decay=1e-5,
        physics_weight=0.1,
        boundary_weight=0.1,
        conservation_weight=0.1,
        settling_velocities=None,
        include_coordinates=True,  # Whether to include lat/lon - geography matters
    ):
        """
        Physics-Informed Neural Network (PINN) for plastic transport modeling.

        This model combines a Fourier Neural Operator with physics constraints to predict
        plastic concentration evolution while respecting conservation laws.

        Think of it as teaching a neural network about fluid dynamics without the PhD.
        """
        super(PINNModel, self).__init__()

        self.save_hyperparameters()

        self.fno_model = FNOModel(
            in_channels=in_channels,
            out_channels=out_channels,
            modes1=modes1,
            modes2=modes2,
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
        Shapes:
            x: [batch_size, in_channels, height, width]
            coords: [batch_size, 2, height, width] (Optional)

        Returns:
            [batch_size, out_channels, height, width]

        """
        return self.fno_model(x, coords)

    def compute_physics_loss(self, x, y_pred, y_true, cell_area, dt=1.0):
        # FIXME: How did the plastic emission and deposition actually modeled in the paper?
        """
        The physics includes:

        - Advection: Wind pushing plastic around (div(u * y))
        - Settling: Gravity pulling plastic down (w_s * y)
        - Conservation: Plastic doesn't magically appear/disappear, or is it? 🤨
        - Boundary conditions: No plastic escaping the domain
        - Mass conservation: Total plastic should remain constant, kind of?

        Args:
            x: [batch_size, in_channels, height, width]
            y_pred: [batch_size, out_channels, height, width]
            y_true: [batch_size, out_channels, height, width]
            cell_area: [batch_size, height, width]
            dt: (default: 1.0)

        Returns:
            Dictionary of physics-informed loss terms for each size bin
        """
        device = y_pred.device

        # Extract relevant variables from input tensor
        # ps = x[:, 0:1]  # Surface pressure - the weight of the atmosphere
        u = x[:, 1:2]  # Zonal wind (x) - eastward/westward flow
        v = x[:, 2:3]  # Meridional wind (y) - northward/southward flow
        # t = x[:, 3:4]  # Temperature - how spicy the air is
        # q = x[:, 4:5]  # Specific humidity - how wet the air is
        # trop_lev = x[:, 5:6]  # Tropopause level - ceiling of the weather

        # Extract emissions - the plastic source terms
        emissions = x[:, -self.hparams.out_channels :]

        # Calculate spatial gradients for each plastic size bin
        dx_list = []
        dy_list = []

        for i in range(self.hparams.out_channels):
            conc = y_pred[:, [i]]

            # Calculate x-gradient with central differencing
            dx = torch.zeros_like(conc)
            dx[..., 1:-1] = (conc[..., 2:] - conc[..., :-2]) / 2

            # Handle periodic boundary in longitude (east-west wraps around)
            dx[..., 0] = (conc[..., 1] - conc[..., -1]) / 2
            dx[..., -1] = (conc[..., 0] - conc[..., -2]) / 2

            # Calculate y-gradient with central differencing
            dy = torch.zeros_like(conc)
            dy[..., 1:-1, :] = (conc[..., 2:, :] - conc[..., :-2, :]) / 2

            # Forward diff at south pole
            dy[..., 0, :] = conc[..., 1, :] - conc[..., 0, :]

            # Backward diff at north pole
            dy[..., -1, :] = conc[..., -1, :] - conc[..., -2, :]

            dx_list.append(dx)
            dy_list.append(dy)

        dx = torch.cat(dx_list, dim=1)
        dy = torch.cat(dy_list, dim=1)

        advection = u * dx + v * dy

        settling_vel = self.settling_vel.to(device).view(1, -1, 1, 1)

        settling_term = settling_vel * y_pred

        # Pre-write in case I decided to add time-dependence
        time_derivative = (y_pred - y_true) / dt

        conservation_residual = time_derivative + advection - emissions + settling_term

        conservation_loss = torch.mean(conservation_residual**2)

        # No plastic at domain edges (simplified)
        boundary_loss = torch.mean(y_pred[:, :, 0, :] ** 2)

        # Mass conservation loss
        # The issue is here - cell_area needs to be properly broadcast to match y_pred's dimensions
        # cell_area shape: [batch_size, height, width]
        # y_pred shape: [batch_size, out_channels, height, width]

        # Expand cell_area to match y_pred's dimensions
        cell_area_expanded = cell_area.unsqueeze(1).expand_as(y_pred)

        mass_pred = torch.sum(y_pred * cell_area_expanded, dim=(-2, -1))
        mass_true = torch.sum(y_true * cell_area_expanded, dim=(-2, -1))

        mass_conservation_loss = torch.mean((mass_pred - mass_true) ** 2)

        return {
            "conservation_loss": conservation_loss,
            "boundary_loss": boundary_loss,
            "mass_conservation_loss": mass_conservation_loss,
        }

    def compute_nn_loss(self, x, y_pred, y_true, cell_area):
        """
        Shapes:
            x: [batch_size, in_channels, height, width]
            y_pred: [batch_size, out_channels, height, width]
            y_true: [batch_size, out_channels, height, width]
            cell_area: [batch_size, height, width]

        Returns:
            Tuple of (physics_losses, data_loss, total_loss)
        """
        pi_losses = self.compute_physics_loss(x, y_pred, y_true, cell_area)

        nn_losses = F.mse_loss(y_pred, y_true)

        pinn_losses = (
            nn_losses
            + self.hparams.physics_weight * pi_losses["conservation_loss"]
            + self.hparams.boundary_weight * pi_losses["boundary_loss"]
            + self.hparams.conservation_weight * pi_losses["mass_conservation_loss"]
        )

        return pi_losses, nn_losses, pinn_losses

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x = batch["inputs"]
        y_true = batch["targets"]
        coords = batch["coords"]
        cell_area = batch["cell_area"]

        y_pred = self(x, coords)

        pi_losses, nn_losses, pinn_losses = self.compute_nn_loss(
            x, y_pred, y_true, cell_area
        )

        self.log("train_data_loss", nn_losses)
        self.log("train_conservation_loss", pi_losses["conservation_loss"])
        self.log("train_boundary_loss", pi_losses["boundary_loss"])
        self.log("train_mass_conservation_loss", pi_losses["mass_conservation_loss"])
        self.log("train_total_loss", pinn_losses)

        return pinn_losses

    def validation_step(self, batch, batch_idx) -> torch.Tensor:

        x = batch["inputs"]
        y_true = batch["targets"]
        coords = batch["coords"]
        cell_area = batch["cell_area"]

        y_pred = self(x, coords)

        pi_losses, nn_losses, pinn_losses = self.compute_nn_loss(
            x, y_pred, y_true, cell_area
        )

        self.log("val_data_loss", nn_losses)
        self.log("val_conservation_loss", pi_losses["conservation_loss"])
        self.log("val_boundary_loss", pi_losses["boundary_loss"])
        self.log("val_mass_conservation_loss", pi_losses["mass_conservation_loss"])
        self.log("val_total_loss", pinn_losses)

        return pinn_losses

    def test_step(self, batch, batch_idx) -> torch.Tensor:

        x = batch["inputs"]
        y_true = batch["targets"]
        coords = batch["coords"]
        cell_area = batch["cell_area"]

        y_pred = self(x, coords)

        pi_losses, nn_losses, pinn_losses = self.compute_nn_loss(
            x, y_pred, y_true, cell_area
        )

        self.log("test_data_loss", nn_losses)
        self.log("test_conservation_loss", pi_losses["conservation_loss"])
        self.log("test_boundary_loss", pi_losses["boundary_loss"])
        self.log("test_mass_conservation_loss", pi_losses["mass_conservation_loss"])
        self.log("test_total_loss", pinn_losses)

        return pinn_losses

    def configure_optimizers(self):
        # TODO: Default optimizer for now. There should be a specilise on for PINN?
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
