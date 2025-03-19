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
        in_channels=7,  # Order: PS, (wind)U, (wind)V, T, Q, TROPLEV, emissions
        out_channels=18,  # 6 sizes × 3 (MMR, DryDep, WetDep)
        modes1=12,
        modes2=12,
        width=32,
        num_layers=2,
        learning_rate=1e-3,
        weight_decay=1e-5,
        physics_weight=0.1,
        boundary_weight=0.1,
        conservation_weight=0.1,
        deposition_weight=0.1,
        settling_velocities=None,
        include_coordinates=True,
    ):
        """
        Extended PINN model to predict MMR, Dry Deposition, and Wet Deposition.
        """
        super(PINNModel, self).__init__()

        self.save_hyperparameters()

        # No. size bins
        self.n_particles = 6

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
        Forward pass returning MMR and deposition predictions.

        Returns:
            tuple: (mmr, dry_dep, wet_dep) each of shape [batch_size, n_particles, height, width]
        """
        outputs = self.fno_model(x, coords)

        # Split the outputs into MMR and depositions
        # Each has n_particles channels
        mmr = outputs[:, : self.n_particles]
        dry_dep = outputs[:, self.n_particles : 2 * self.n_particles]
        wet_dep = outputs[:, 2 * self.n_particles :]

        return mmr, dry_dep, wet_dep

    def compute_physics_loss(self, x, predictions, targets, cell_area, dt=1.0):
        """
        Compute physics-informed losses for MMR and deposition predictions.

        Args:
            x: Input features [batch_size, in_channels, height, width]
            predictions: Tuple of (mmr_pred, dry_dep_pred, wet_dep_pred)
            targets: Tuple of (mmr_true, dry_dep_true, wet_dep_true)
            cell_area: Area of each grid cell [height, width]
            dt: Time step (right now we don't do time-dependent simulations)
        """
        mmr_pred, dry_dep_pred, wet_dep_pred = predictions
        mmr_true, dry_dep_true, wet_dep_true = targets

        batch_size, n_particles, height, width = mmr_pred.shape
        device = mmr_pred.device

        # Meteorological
        ps = x[:, 0:1]
        u = x[:, 1:2]
        v = x[:, 2:3]
        emissions = x[:, -1:]

        # Settling velocity
        v_s = (
            self.settling_vel.view(1, -1, 1, 1)
            .expand(batch_size, -1, height, width)
            .to(device)
        )

        losses = {}

        # L_mass = (∫∫∫ ρ·MMR dV + ∫∫ (DryDep + WetDep) dA - ∫∫ Emissions dA)²

        mass_true = (mmr_true * cell_area[None, None, :, :]).sum(dim=(-2, -1))
        mass_pred = (mmr_pred * cell_area[None, None, :, :]).sum(dim=(-2, -1))
        dry_dep = (dry_dep_pred * cell_area[None, None, :, :]).sum(dim=(-2, -1))
        wet_dep = (wet_dep_pred * cell_area[None, None, :, :]).sum(dim=(-2, -1))
        emissions = (emissions * cell_area[None, None, :, :]).sum(dim=(-2, -1))

        mass_conservation_error = mass_pred - mass_true + dry_dep + wet_dep - emissions
        losses["mass_conservation"] = torch.mean(mass_conservation_error**2)

        # L_transport = ∫∫∫ (∂C/∂t + u·∂C/∂x + v·∂C/∂y + w·∂C/∂z - K_h∇²C - K_v∂²C/∂z² - S)² dV dt
        # NOTE: # Central differences for interior points

        dC_dt = (mmr_pred - mmr_true) / dt

        # Spatial derivatives (central differences)
        # Pad for boundary conditions (periodic assumed for simplicity)
        mmr_pad = F.pad(mmr_pred, (1, 1, 1, 1), mode="circular")

        # Assuming dx and dy are grid spacings (could be passed as parameters)
        # For simplicity, using normalized grid spacing
        dx, dy = 1.0, 1.0

        dC_dx = (mmr_pad[:, :, 1:-1, 2:] - mmr_pad[:, :, 1:-1, :-2]) / (2 * dx)
        dC_dy = (mmr_pad[:, :, 2:, 1:-1] - mmr_pad[:, :, :-2, 1:-1]) / (2 * dy)

        # TODO: w term and diffusion terms

        # Simplified transport residual without diffusion terms
        # (∂C/∂t + u·∂C/∂x + v·∂C/∂y - S)
        transport_residual = dC_dt + u * dC_dx + v * dC_dy - emissions

        # Compute transport
        losses["transport"] = torch.mean(transport_residual**2)

        # # Questionable. Settling velocity should be correlated with deposition rate.
        # settling_dep_correlation = v_s * mmr_pred - dry_dep_pred
        # losses["settling_deposition"] = torch.mean(settling_dep_correlation**2)

        # # Questionable. Physical relationship between MMR and deposition rates
        # total_dep = dry_dep_pred + wet_dep_pred
        # dep_balance = torch.mean((total_dep - v_s * mmr_pred) ** 2)
        # losses["deposition_balance"] = dep_balance

        # # Questionable. All quantities should be non-negative
        # non_negative_loss = (
        #     torch.mean(F.relu(-mmr_pred))
        #     + torch.mean(F.relu(-dry_dep_pred))
        #     + torch.mean(F.relu(-wet_dep_pred))
        # )
        # losses["non_negative"] = non_negative_loss

        # Combine all physics losses
        total_physics_loss = (
            self.hparams.conservation_weight * losses["mass_conservation"]
            + self.hparams.physics_weight * losses["settling_deposition"]
            + self.hparams.deposition_weight * losses["deposition_balance"]
            + 0.1 * losses["non_negative"]  # Small weight for non-negativity
        )

        losses["total_physics"] = total_physics_loss
        return losses

    def compute_loss(self, x, predictions, targets, cell_area):
        """
        Compute total loss combining MSE and physics-informed losses.
        """
        mmr_pred, dry_dep_pred, wet_dep_pred = predictions
        mmr_true, dry_dep_true, wet_dep_true = targets

        # Data fidelity losses for each output
        mse_mmr = F.mse_loss(mmr_pred, mmr_true)
        mse_dry = F.mse_loss(dry_dep_pred, dry_dep_true)
        mse_wet = F.mse_loss(wet_dep_pred, wet_dep_true)

        # Combined MSE loss
        mse_loss = mse_mmr + mse_dry + mse_wet

        # Physics-informed losses
        physics_losses = self.compute_physics_loss(x, predictions, targets, cell_area)

        # Combine losses
        total_loss = mse_loss + physics_losses["total_physics"]

        # Create loss dictionary
        losses = {
            "mse_mmr": mse_mmr,
            "mse_dry_dep": mse_dry,
            "mse_wet_dep": mse_wet,
            "mse_total": mse_loss,
            "total_loss": total_loss,
            **physics_losses,
        }

        return losses

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Training step handling multiple outputs.
        """
        x, targets, cell_area = batch
        mmr_true, dry_dep_true, wet_dep_true = targets  # Unpack targets

        # Forward pass
        predictions = self(x)  # Returns (mmr_pred, dry_dep_pred, wet_dep_pred)

        # Compute losses
        losses = self.compute_loss(x, predictions, targets, cell_area)

        # Log all losses
        for name, value in losses.items():
            self.log(f"train_{name}", value, prog_bar=name == "total_loss")

        return losses["total_loss"]

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Perform a validation step.

        Args:
            batch: Batch of data containing inputs, targets, and cell areas
            batch_idx: Index of the batch

        Returns:
            Total validation loss
        """
        x, y_true, cell_area = batch

        # Forward pass
        y_pred = self(x)

        # Compute losses
        losses = self.compute_loss(x, y_pred, y_true, cell_area)

        # Log all losses
        for name, value in losses.items():
            self.log(f"val_{name}", value, prog_bar=name == "total_loss")

        return losses["total_loss"]

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Perform a test step.

        Args:
            batch: Batch of data containing inputs, targets, and cell areas
            batch_idx: Index of the batch

        Returns:
            Total test loss
        """
        x, y_true, cell_area = batch

        # Forward pass
        y_pred = self(x)

        # Compute losses
        losses = self.compute_loss(x, y_pred, y_true, cell_area)

        # Calculate additional metrics for evaluation
        rmse = torch.sqrt(F.mse_loss(y_pred, y_true))

        # Log all losses and metrics
        for name, value in losses.items():
            self.log(f"test_{name}", value)

        self.log("test_rmse", rmse)

        return losses["total_loss"]

    def configure_optimizers(self):
        # TODO: Default optimizer for now. There should be a specilised one for PINN?
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
