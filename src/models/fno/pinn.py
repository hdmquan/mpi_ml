import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from loguru import logger

from src.models.fno.core import FNOModel
from src.utils import set_seed

set_seed()




class PINNModel(pl.LightningModule):

    def __init__(
        self,
        # Order: PS, (wind)U, (wind)V, T, Q, TROPLEV, emissions
        # Translate: Pressure, Wind East, Wind North, Temperature, Humidity, Troposphere, Source term
        in_channels=7,  
        out_channels=18,  # 6 sizes × 3 (MMR, DryDep, WetDep)
        # FNO settings
        modes1=12,
        modes2=12,
        width=32,
        num_layers=2,
        learning_rate=1e-3,
        weight_decay=1e-5,
        # Downplay PINN lossed for now
        mmr_weight=0.1,
        deposition_weight=0.1,
        conservation_weight=0.1,
        physics_weight=0.1,
        boundary_weight=0.1,
        # 
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
        if coords is None:
            coords = torch.zeros_like(x[:, :2, :, :])

        outputs = self.fno_model(x, coords)

        mmr = outputs[:, : self.n_particles]
        dry_dep = outputs[:, self.n_particles : 2 * self.n_particles]
        wet_dep = outputs[:, 2 * self.n_particles :]

        return mmr, dry_dep, wet_dep

    def compute_loss(self, x, predictions, targets, cell_area, dt=1.0):
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

        ps = x[:, [0]]
        u = x[:, [1]]
        v = x[:, [2]]
        emissions = x[:, [-1]]

        # logger.info("Initial shapes")

        # logger.debug(f"Pressure: {ps.shape}")
        # logger.debug(f"Wind East: {u.shape}")
        # logger.debug(f"Wind North: {v.shape}")
        # logger.debug(f"Emissions: {emissions.shape}")

        # Settling velocity
        v_s = (
            self.settling_vel.view(1, -1, 1, 1)
            .expand(batch_size, -1, height, width)
            .to(device)
        )

        # logger.debug(f"Settling velocity: {v_s.shape}")

        losses = {}

        # L_mmr = (1/N) Σ (MMR pred - MMR true)^2
        losses["mmr"] = F.mse_loss(mmr_pred, mmr_true)

        # L_dep = (1/N) Σ (DryDep pred - DryDep true)^2 + (1/N) Σ (WetDep pred - WetDep true)^2
        losses["deposition"] = F.mse_loss(dry_dep_pred, dry_dep_true) + F.mse_loss(
            wet_dep_pred, wet_dep_true
        )

        # logger.info("Mass conservation loss")

        # L_mass = (∫∫∫ ρ·MMR dV + ∫∫ (DryDep + WetDep) dA - ∫∫ Emissions dA)²
        # Store the original emissions shape before summation
        # ρ is ignore since this is a loss
        emissions_grid = emissions  # Shape: [12, 1, 500, 400]

        # Calculate mass conservation using summed values
        mass_true = (mmr_true * cell_area[None, None, :, :]).sum(dim=(-2, -1))
        mass_pred = (mmr_pred * cell_area[None, None, :, :]).sum(dim=(-2, -1))

        dry_dep = (dry_dep_pred * cell_area[None, None, :, :]).sum(dim=(-2, -1))
        wet_dep = (wet_dep_pred * cell_area[None, None, :, :]).sum(dim=(-2, -1))

        emissions_sum = (emissions * cell_area[None, None, :, :]).sum(dim=(-2, -1))

        # logger.debug(f"Mass true: {mass_true.shape}")
        # logger.debug(f"Mass pred: {mass_pred.shape}")
        # logger.debug(f"Dry deposition: {dry_dep.shape}")
        # logger.debug(f"Wet deposition: {wet_dep.shape}")
        # logger.debug(f"Emissions sum: {emissions_sum.shape}")

        mass_conservation_error = (
            (mass_pred - mass_true) + dry_dep + wet_dep - emissions_sum
        )
        total_mass = torch.abs(mass_true).mean() + torch.abs(emissions_sum).mean()

        # Normalize since it's kinda dominate others
        losses["mass_conservation"] = torch.mean(
            (mass_conservation_error / (total_mass + 1e-8)) ** 2
        )

        # logger.info("Transport loss")

        # L_transport = ∫∫∫ (∂C/∂t + u·∂C/∂x + v·∂C/∂y + w·∂C/∂z - K_h∇²C - K_v∂²C/∂z² - S)² dV dt
        dC_dt = (mmr_pred - mmr_true) / dt

        # Spatial derivatives (central differences)
        # Pad for boundary conditions (periodic assumed for simplicity)
        mmr_pad = F.pad(mmr_pred, (1, 1, 1, 1), mode="circular")

        # Assuming dx and dy are grid spacings (could be passed as parameters)
        # For simplicity, using normalized grid spacing
        dx, dy = 1.0, 1.0

        dC_dx = (mmr_pad[:, :, 1:-1, 2:] - mmr_pad[:, :, 1:-1, :-2]) / (2 * dx)
        dC_dy = (mmr_pad[:, :, 2:, 1:-1] - mmr_pad[:, :, :-2, 1:-1]) / (2 * dy)

        # logger.debug(f"MMR pad: {mmr_pad.shape}")
        # logger.debug(f"dC_dt: {dC_dt.shape}")
        # logger.debug(f"dC_dx: {dC_dx.shape}")
        # logger.debug(f"dC_dy: {dC_dy.shape}")

        # Note: w·∂C/∂z and K_v∂²C/∂z² terms are not implemented because
        # the current model is 2D (no vertical dimension)
        # For a full 3D implementation, we would need:
        # dC_dz = (mmr_pad[:, :, 2:, 1:-1, 1:-1] - mmr_pad[:, :, :-2, 1:-1, 1:-1]) / (2 * dz)
        # We leave these as 0 for now

        # Horizontal diffusion term (Laplace)
        d2C_dx2 = (
            mmr_pad[:, :, 1:-1, 2:] - 2 * mmr_pred + mmr_pad[:, :, 1:-1, :-2]
        ) / (dx**2)

        d2C_dy2 = (
            mmr_pad[:, :, 2:, 1:-1] - 2 * mmr_pred + mmr_pad[:, :, :-2, 1:-1]
        ) / (dy**2)

        laplacian_C = d2C_dx2 + d2C_dy2

        # logger.debug(f"Laplacian: {laplacian_C.shape}")

        # Assuming Kh = 1.0 for simplicity. Should it be here since this is a loss?
        Kh = 1.0
        diffusion_term = Kh * laplacian_C

        # logger.debug(f"diffusion_term: {diffusion_term.shape}")
        # logger.debug(f"emissions: {emissions.shape}")

        # Use emissions_grid instead of emissions for transport residual
        # Expand emissions_grid to match particle dimension
        emissions_grid = emissions_grid.expand(-1, mmr_pred.size(1), -1, -1)
        transport_residual = (
            dC_dt + u * dC_dx + v * dC_dy - diffusion_term - emissions_grid
        )
        losses["transport"] = torch.mean(transport_residual**2)

        # L_settling = (1/V) ∫∫∫ (∂C/∂t + v_s·∂C/∂z)² dV dt
        # Note: Since we're in 2D, we approximate this with the vertical settling effect
        # In a full 3D model, we would compute ∂C/∂z correctly
        # For now, we use a proxy based on the settling velocity and MMR
        settling_residual = dC_dt + v_s * mmr_pred
        losses["settling"] = torch.mean(settling_residual**2)

        # L_boundary = (1/A) ∫∫ (v_s·C_surface - (DryDep + Emissions))² dA dt
        # For 2D, we consider the entire domain as the surface
        boundary_residual = v_s * mmr_pred - (dry_dep_pred + emissions)
        losses["boundary"] = torch.mean(boundary_residual**2)

        # L_smooth = λ_s ∫∫∫ |∇C|² dV
        # Gradient magnitude squared
        gradient_mag_sq = dC_dx**2 + dC_dy**2
        smoothness_param = 0.01  # λ_s
        losses["smoothness"] = smoothness_param * torch.mean(gradient_mag_sq)

        # Questionable. Settling velocity should be correlated with deposition rate.
        settling_dep_correlation = v_s * mmr_pred - dry_dep_pred
        losses["settling_deposition"] = torch.mean(settling_dep_correlation**2)

        # Questionable. Physical relationship between MMR and deposition rates
        total_dep = dry_dep_pred + wet_dep_pred
        dep_balance = torch.mean((total_dep - v_s * mmr_pred) ** 2)
        losses["deposition_balance"] = dep_balance

        # Questionable. Since I deliberately used Softplus?
        non_negative_loss = (
            torch.mean(F.relu(-mmr_pred))
            + torch.mean(F.relu(-dry_dep_pred))
            + torch.mean(F.relu(-wet_dep_pred))
        )
        losses["non_negative"] = non_negative_loss

        total_loss = (
            self.hparams.mmr_weight * losses["mmr"]
            + self.hparams.deposition_weight * losses["deposition"]
            + self.hparams.conservation_weight * losses["mass_conservation"]
            + self.hparams.physics_weight * losses["transport"]
            + self.hparams.physics_weight * losses["settling"]
            + self.hparams.boundary_weight * losses["boundary"]
            + self.hparams.physics_weight * losses["smoothness"]
            + self.hparams.physics_weight * losses["settling_deposition"]
            + self.hparams.deposition_weight * losses["deposition_balance"]
            + 0.1 * losses["non_negative"]
        )

        # logger.info("Losses")
        # logger.debug(f"Mass conservation: {losses['mass_conservation']}")
        # logger.debug(f"Transport: {losses['transport']}")
        # logger.debug(f"Settling: {losses['settling']}")
        # logger.debug(f"Boundary: {losses['boundary']}")
        # logger.debug(f"Smoothness: {losses['smoothness']}")
        # logger.debug(f"Settling deposition: {losses['settling_deposition']}")
        # logger.debug(f"Deposition balance: {losses['deposition_balance']}")
        # logger.debug(f"Non-negative: {losses['non_negative']}")
        # logger.debug(f"Total loss: {total_loss}")

        losses["total"] = total_loss
        return losses

    def training_step(self, batch, batch_idx) -> torch.Tensor:

        x, targets, cell_area = batch

        predictions = self(x)
        losses = self.compute_loss(x, predictions, targets, cell_area)

        for name, value in losses.items():
            self.log(f"train_{name}", value, prog_bar=name == "total_loss")

        return losses["total_loss"]

    def validation_step(self, batch, batch_idx) -> torch.Tensor:

        x, y_true, cell_area = batch

        y_pred = self(x)
        losses = self.compute_loss(x, y_pred, y_true, cell_area)

        for name, value in losses.items():
            self.log(f"val_{name}", value, prog_bar=name == "total_loss")

        return losses["total_loss"]

    def test_step(self, batch, batch_idx) -> torch.Tensor:

        x, y_true, cell_area = batch

        y_pred = self(x)
        losses = self.compute_loss(x, y_pred, y_true, cell_area)

        rmse = torch.sqrt(F.mse_loss(y_pred, y_true))

        for name, value in losses.items():
            self.log(f"test_{name}", value)

        self.log("test_rmse", rmse)

        return losses["total_loss"]

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
    model = PINNModel()

    x = torch.randn(12, 7, 500, 400, requires_grad=True)

    mmr, dry_dep, wet_dep = model(x)

    sample_mmr = torch.randn(12, 6, 48, 500, 400)
    sample_dry_dep = torch.randn(12, 6, 500, 400)
    sample_wet_dep = torch.randn(12, 6, 500, 400)

    # Sample loss
    losses = model.compute_loss(
        x,
        (sample_mmr, sample_dry_dep, sample_wet_dep),
        (mmr, dry_dep, wet_dep),
        torch.ones_like(mmr),
    )

    losses["total"].backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.isnan(
            param.grad
        ).any(), f"Parameter {name} has NaN gradient values"
        assert not torch.isinf(
            param.grad
        ).any(), f"Parameter {name} has Inf gradient values"

    logger.success("Gradient check passed successfully!")
