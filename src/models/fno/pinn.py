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
        """
        Compute physics-informed losses based on conservation laws and transport equations.

        Args:
            x: Input features [batch_size, in_channels, height, width]
            y_pred: Predicted mass mixing ratio [batch_size, out_channels, height, width]
            y_true: Ground truth mass mixing ratio [batch_size, out_channels, height, width]
            cell_area: Area of each grid cell [height, width]
            dt: Time step between consecutive frames

        Returns:
            Dictionary of individual physics losses and their weighted sum
        """
        batch_size, n_particles, height, width = y_pred.shape
        device = y_pred.device

        # Extract relevant input fields (assuming standard order)
        ps = x[:, 0:1]  # Surface pressure
        u = x[:, 1:2]  # Eastward wind
        v = x[:, 2:3]  # Northward wind
        # w = x[:, 3:4]  # Vertical velocity (if available)
        emissions = x[:, -1:]  # Assuming emissions is the last channel

        # Reshaping settling velocities for broadcasting
        # [6] -> [batch, 6, height, width]
        v_s = (
            self.settling_vel.view(1, -1, 1, 1)
            .expand(batch_size, -1, height, width)
            .to(device)
        )

        losses = {}

        # 1. Mass Conservation Loss
        # L_mass = (∫∫∫ ρ·MMR dV + ∫∫ (DryDep + WetDep) dA - ∫∫ Emissions dA)²
        # Simplified: For now, we check if the total mass remains constant minus emissions
        total_mass_t0 = (y_true[:, :, :, :] * cell_area[None, None, :, :]).sum(
            dim=(-2, -1)
        )
        total_mass_t1 = (y_pred[:, :, :, :] * cell_area[None, None, :, :]).sum(
            dim=(-2, -1)
        )
        total_emissions = (emissions * cell_area[None, None, :, :]).sum(dim=(-2, -1))

        # Simplified mass conservation: change in mass should equal emissions
        mass_conservation_error = total_mass_t1 - total_mass_t0 - total_emissions
        losses["mass_conservation"] = torch.mean(mass_conservation_error**2)

        # 2. Transport Equation Loss
        # L_transport = ∫∫∫ (∂C/∂t + u·∂C/∂x + v·∂C/∂y + w·∂C/∂z - K_h∇²C - K_v∂²C/∂z² - S)² dV dt

        # Compute gradients using finite differences
        # Note: requires spatial coordinates to be evenly spaced for accurate derivatives
        # We'll use central differences for interior points

        # Temporal derivative (forward difference)
        dC_dt = (y_pred - y_true) / dt

        # Spatial derivatives (central differences)
        # Pad for boundary conditions (periodic assumed for simplicity)
        y_pad = F.pad(y_pred, (1, 1, 1, 1), mode="circular")

        # Assuming dx and dy are grid spacings (could be passed as parameters)
        # For simplicity, using normalized grid spacing
        dx, dy = 1.0, 1.0

        # Compute x-derivatives using central differences
        dC_dx = (y_pad[:, :, 1:-1, 2:] - y_pad[:, :, 1:-1, :-2]) / (2 * dx)

        # Compute y-derivatives using central differences
        dC_dy = (y_pad[:, :, 2:, 1:-1] - y_pad[:, :, :-2, 1:-1]) / (2 * dy)

        # Simplified transport residual without diffusion terms
        # (∂C/∂t + u·∂C/∂x + v·∂C/∂y - S)
        transport_residual = dC_dt + u * dC_dx + v * dC_dy - emissions

        # Compute transport loss
        losses["transport"] = torch.mean(transport_residual**2)

        # 3. Settling Velocity Loss
        # L_settling = ∫∫∫ (∂C/∂t + v_s·∂C/∂z)² dV dt
        # Note: This is simplified without vertical info, but we can penalize based on settling
        # Different particle sizes have different settling behaviors

        # Simple settling loss (approximation without vertical dimension)
        # Higher settling velocity should lead to faster deposition
        settling_effect = v_s * y_pred  # Proxy for settling effect

        # The settling loss ensures particles with higher settling velocity have stronger
        # downward tendency (would need vertical dimension for full implementation)
        losses["settling"] = torch.mean((dC_dt + settling_effect) ** 2)

        # 4. Boundary Layer Interaction Loss
        # L_boundary = ∫∫ (v_s·C_surface - (DryDep + Emissions))² dA dt
        # Simplified since we don't have explicit deposition in this prediction
        boundary_residual = v_s * y_pred - emissions
        losses["boundary"] = torch.mean(boundary_residual**2)

        # 5. Smoothness Regularization
        # L_smooth = λ_s ∫∫∫ |∇C|² dV
        # Gradient magnitude squared
        gradient_magnitude_squared = dC_dx**2 + dC_dy**2
        losses["smoothness"] = torch.mean(gradient_magnitude_squared)

        # Combine all physics losses with weights
        total_physics_loss = (
            self.hparams.conservation_weight * losses["mass_conservation"]
            + self.hparams.physics_weight * losses["transport"]
            + self.hparams.physics_weight * losses["settling"]
            + self.hparams.boundary_weight * losses["boundary"]
            + 0.01 * losses["smoothness"]  # Small weight for smoothness
        )

        losses["total_physics"] = total_physics_loss
        return losses

    def compute_loss(self, x, y_pred, y_true, cell_area):
        """
        Compute the total loss combining data fidelity (MSE) and physics-informed losses.

        Args:
            x: Input features [batch_size, in_channels, height, width]
            y_pred: Predicted mass mixing ratio [batch_size, out_channels, height, width]
            y_true: Ground truth mass mixing ratio [batch_size, out_channels, height, width]
            cell_area: Area of each grid cell [height, width]

        Returns:
            Dictionary of losses including the total loss
        """
        # Data fidelity loss (MSE)
        mse_loss = F.mse_loss(y_pred, y_true)

        # Physics-informed losses
        physics_losses = self.compute_physics_loss(x, y_pred, y_true, cell_area)

        # Combine losses
        total_loss = mse_loss + physics_losses["total_physics"]

        # Create loss dictionary
        losses = {
            "mse": mse_loss,
            "total_loss": total_loss,
            **physics_losses,  # Include all individual physics losses
        }

        return losses

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Perform a training step.

        Args:
            batch: Batch of data containing inputs, targets, and cell areas
            batch_idx: Index of the batch

        Returns:
            Total loss for backpropagation
        """
        x, y_true, cell_area = batch

        # Forward pass
        y_pred = self(x)

        # Compute losses
        losses = self.compute_loss(x, y_pred, y_true, cell_area)

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
