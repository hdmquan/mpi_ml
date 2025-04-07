import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from abc import ABC, abstractmethod
import torch.nn as nn
from loguru import logger

from src.utils import set_seed
from src.data.module import get_normalization_stats

set_seed()


class Base(pl.LightningModule, ABC):

    def __init__(
        self,
        in_channels=7,
        out_channels=6,
        transport_loss_weight=1.0,
        settling_velocities=None,
        include_coordinates=True,
        use_physics_loss=False,
        use_mass_conservation_loss=False,
        mass_conservation_weight=0.1,
        learning_rate=1e-4,
        weight_decay=1e-5,
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
        x, targets, metadata = batch
        predictions = self(x)
        losses = self.compute_loss(x, predictions, targets, metadata)

        # Log all loss components
        for name, value in losses.items():
            self.log(f"train_{name}", value, prog_bar=name == "total")

        return losses["total"]

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x, y_true, metadata = batch
        y_pred = self(x)
        losses = self.compute_loss(x, y_pred, y_true, metadata)

        # Log all loss components
        for name, value in losses.items():
            self.log(f"val_{name}", value, prog_bar=name == "total")

        return losses["total"]

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        x, y_true, metadata = batch
        y_pred = self(x)
        losses = self.compute_loss(x, y_pred, y_true, metadata)
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

    def compute_loss(self, x, mmr_pred, mmr_true, metadata=None):
        losses = {}
        # Traditional MSE loss
        mse_losses = self.compute_traditional_loss(mmr_pred, mmr_true)
        losses.update(mse_losses)

        # Physics-based loss if enabled
        if self.hparams.use_physics_loss:
            transport_loss = self.compute_transport_loss(x, mmr_pred)
            losses["transport"] = transport_loss
            losses["total"] = (
                mse_losses["total"]
                + self.hparams.transport_loss_weight * transport_loss
            )
        else:
            losses["total"] = mse_losses["total"]

        # Mass conservation loss if enabled and metadata is provided
        if self.hparams.use_mass_conservation_loss and metadata is not None:
            try:
                norm_stats = get_normalization_stats(self.trainer.datamodule.h5_file)
                mass_losses = self.compute_mass_loss(
                    x, mmr_pred, mmr_true, metadata, norm_stats
                )
                losses.update(mass_losses)
                losses["total"] += (
                    self.hparams.mass_conservation_weight
                    * mass_losses["mass_conservation"]
                )
            except Exception as e:
                logger.warning(f"Failed to compute mass conservation loss: {e}")

        return losses

    def compute_traditional_loss(self, pred, true):
        # Shape [1, 6, 50, 384, 576]
        # MMR [1, 6, :-2, 384, 576]
        # Dry + Wet Deposition [1, -2:, 50, 384, 576]

        mmr_pred = pred[:, :, :-2, :, :]
        dry_dep_pred = pred[:, :, -2, :, :]
        wet_dep_pred = pred[:, :, -1, :, :]

        mmr_true = true[:, :, :-2, :, :]
        dry_dep_true = true[:, :, -2, :, :]
        wet_dep_true = true[:, :, -1, :, :]

        # print(pred.shape, true.shape)

        # Altitude weights
        # Decrease exponentially as altitude increases
        # From 2 at the surface to 1 at the top
        alt_dim = mmr_pred.shape[2]
        weights = torch.exp(-torch.arange(alt_dim, device=pred.device) * 0.5)
        weights = 1 + (weights - weights.min()) / (weights.max() - weights.min())
        weights = weights.view(1, 1, -1, 1, 1)

        # Apply weights to squared differences
        print(mmr_pred.shape, mmr_true.shape)
        mmr_squared_diff = (mmr_pred - mmr_true) ** 2
        mmr_weighted_squared_diff = mmr_squared_diff * weights
        mmr_loss = mmr_weighted_squared_diff.mean()

        # Apply weight of 2 to depositions (because it's more important than higher altitudes mmr)
        deposition_loss = 2.0 * (
            F.mse_loss(dry_dep_pred, dry_dep_true)
            + F.mse_loss(wet_dep_pred, wet_dep_true)
        )

        total_loss = mmr_loss + deposition_loss

        return {"total": total_loss, "mmr": mmr_loss, "deposition": deposition_loss}

    def compute_transport_loss(self, x, pred):
        # Extract 3D MMR field (excluding deposition layers)
        mmr = pred[:, :, :-2, :, :]

        try:
            batch_size, n_particles, altitude, latitude, longitude = mmr.shape
            device = pred.device

            u = x[:, [1]]  # Wind East
            v = x[:, [2]]  # Wind North
            emissions = x[:, -6:]  # Emissions for each particle size

            v_s = (
                self.settling_vel.view(1, -1, 1, 1, 1)
                .expand(batch_size, -1, altitude, latitude, longitude)
                .to(device)
            )

            # Transport loss (advection + diffusion)
            #     u · ∇C            Advection (transport by wind)
            #   - ∇ · (D ∇C)        Diffusion (spread by turbulent mixing)
            #   + vs * ∂C/∂z        Settling (falling by gravity)
            #   = S                 Source

            # Create coordinate grids
            z = torch.arange(altitude, device=device).float()
            y = torch.arange(latitude, device=device).float()
            x_grid = torch.arange(longitude, device=device).float()

            # Expand to match the shape of mmr
            z = z.view(1, 1, -1, 1, 1).expand_as(mmr).clone()
            y = y.view(1, 1, 1, -1, 1).expand_as(mmr).clone()
            x_grid = x_grid.view(1, 1, 1, 1, -1).expand_as(mmr).clone()

            # Make coordinates require gradients
            z.requires_grad_(True)
            y.requires_grad_(True)
            x_grid.requires_grad_(True)

            # Ensure mmr requires gradients
            mmr = mmr.detach().clone().requires_grad_(True)

            # Compute gradients
            grad_Cz = torch.autograd.grad(
                mmr,
                z,
                grad_outputs=torch.ones_like(mmr),
                create_graph=True,
                allow_unused=True,
            )[0]
            grad_Cy = torch.autograd.grad(
                mmr,
                y,
                grad_outputs=torch.ones_like(mmr),
                create_graph=True,
                allow_unused=True,
            )[0]
            grad_Cx = torch.autograd.grad(
                mmr,
                x_grid,
                grad_outputs=torch.ones_like(mmr),
                create_graph=True,
                allow_unused=True,
            )[0]

            # Replace None gradients with zeros
            if grad_Cz is None:
                grad_Cz = torch.zeros_like(mmr)
            if grad_Cy is None:
                grad_Cy = torch.zeros_like(mmr)
            if grad_Cx is None:
                grad_Cx = torch.zeros_like(mmr)

            # Ensure gradients have requires_grad for the second derivative computation
            if not grad_Cx.requires_grad:
                grad_Cx = grad_Cx.detach().clone().requires_grad_(True)
            if not grad_Cy.requires_grad:
                grad_Cy = grad_Cy.detach().clone().requires_grad_(True)
            if not grad_Cz.requires_grad:
                grad_Cz = grad_Cz.detach().clone().requires_grad_(True)

            # Compute advection
            advection = u * grad_Cx + v * grad_Cy

            # Settling
            settling = v_s * grad_Cz

            # Simplified diffusion term since we are calc loss not simulate
            D = 1.0

            # Compute second derivatives for Laplacian
            laplacian_x = torch.autograd.grad(
                grad_Cx,
                x_grid,
                grad_outputs=torch.ones_like(grad_Cx),
                create_graph=True,
                allow_unused=True,
            )[0]
            laplacian_y = torch.autograd.grad(
                grad_Cy,
                y,
                grad_outputs=torch.ones_like(grad_Cy),
                create_graph=True,
                allow_unused=True,
            )[0]
            laplacian_z = torch.autograd.grad(
                grad_Cz,
                z,
                grad_outputs=torch.ones_like(grad_Cz),
                create_graph=True,
                allow_unused=True,
            )[0]

            # Replace None gradients with zeros
            if laplacian_x is None:
                laplacian_x = torch.zeros_like(grad_Cx)
            if laplacian_y is None:
                laplacian_y = torch.zeros_like(grad_Cy)
            if laplacian_z is None:
                laplacian_z = torch.zeros_like(grad_Cz)

            laplacian = laplacian_x + laplacian_y + laplacian_z
            diffusion = D * laplacian

            # Residual (transport equation)
            residual = advection - diffusion + settling - emissions

            # Transport loss
            return torch.mean(residual**2)

        except RuntimeError as e:
            import traceback

            print(traceback.format_exc())
            # Fallback to a simpler transport loss if we encounter gradient issues
            print(f"Warning: Using simplified transport loss due to error: {e}")

            # Simpler transport loss: just encourage gradients to align with wind directions
            # Extract gradients using finite differences (no autograd needed)
            # These are approximations of the spatial gradients

            # Calculate wind magnitude
            wind_magnitude = torch.sqrt(x[:, 1] ** 2 + x[:, 2] ** 2).unsqueeze(1)

            # Simple advection-only loss using difference along wind direction
            # This is a simpler proxy for the full transport equation
            transport_residual = (
                mmr - emissions.unsqueeze(2)
            ) * wind_magnitude.unsqueeze(2)

            return torch.mean(transport_residual**2)

    def compute_mass_loss(self, x, pred, true, metadata, norm_stats):
        # Split prediction and target into components
        pred_mmr = pred[:, :, :-2, :, :]  # [batch, n_particles, altitude, lat, lon]
        true_mmr = true[:, :, :-2, :, :]
        pred_dry_dep = pred[:, :, -2, :, :]  # [batch, n_particles, lat, lon]
        true_dry_dep = true[:, :, -2, :, :]
        pred_wet_dep = pred[:, :, -1, :, :]
        true_wet_dep = true[:, :, -1, :, :]

        # Denormalize MMR and deposition values efficiently
        mmr_log_scale = norm_stats["mmr_log_scale"]
        mmr_min, mmr_max = mmr_log_scale["min"], mmr_log_scale["max"]
        epsilon = mmr_log_scale["epsilon"]

        # Denormalize MMR (reverse the log-scale normalization)
        # First unnormalize from [0,1] to log scale, then exp to get original values
        pred_mmr_denorm = (
            torch.pow(10, pred_mmr * (mmr_max - mmr_min) + mmr_min) - epsilon
        )
        true_mmr_denorm = (
            torch.pow(10, true_mmr * (mmr_max - mmr_min) + mmr_min) - epsilon
        )

        # Clamp to avoid negative values after denormalization
        pred_mmr_denorm = torch.clamp(pred_mmr_denorm, min=0.0)
        true_mmr_denorm = torch.clamp(true_mmr_denorm, min=0.0)

        # Denormalize depositions (simple linear denormalization)
        # For each particle size
        batch_size, n_particles = pred_mmr.shape[:2]

        # Process all particle sizes at once using broadcasting
        dry_dep_means = torch.tensor(
            [norm_stats["output_mean"]["dry_dep"][i] for i in range(n_particles)],
            device=pred_dry_dep.device,
        ).view(1, n_particles, 1, 1)
        dry_dep_stds = torch.tensor(
            [norm_stats["output_std"]["dry_dep"][i] for i in range(n_particles)],
            device=pred_dry_dep.device,
        ).view(1, n_particles, 1, 1)

        wet_dep_means = torch.tensor(
            [norm_stats["output_mean"]["wet_dep"][i] for i in range(n_particles)],
            device=pred_wet_dep.device,
        ).view(1, n_particles, 1, 1)
        wet_dep_stds = torch.tensor(
            [norm_stats["output_std"]["wet_dep"][i] for i in range(n_particles)],
            device=pred_wet_dep.device,
        ).view(1, n_particles, 1, 1)

        # Denormalize depositions
        pred_dry_dep_denorm = pred_dry_dep * dry_dep_stds + dry_dep_means
        true_dry_dep_denorm = true_dry_dep * dry_dep_stds + dry_dep_means
        pred_wet_dep_denorm = pred_wet_dep * wet_dep_stds + wet_dep_means
        true_wet_dep_denorm = true_wet_dep * wet_dep_stds + wet_dep_means

        # Calculate total mass in the atmosphere using MMR and pressure levels
        pred_atm_mass = self.calculate_mp_mass(
            pred_mmr_denorm,
            metadata["ps"],
            metadata["hyai"],
            metadata["hybi"],
            metadata["gw"],
            metadata["P0"],
        )

        true_atm_mass = self.calculate_mp_mass(
            true_mmr_denorm,
            metadata["ps"],
            metadata["hyai"],
            metadata["hybi"],
            metadata["gw"],
            metadata["P0"],
        )

        # Calculate deposition mass (integrate over surface area)
        cell_area = metadata["gw"].view(1, 1, -1, 1)  # [1, 1, lat, 1]

        # Sum over lat/lon dimensions with cell area weighting
        pred_dry_dep_mass = (pred_dry_dep_denorm * cell_area).sum(dim=(-2, -1))
        true_dry_dep_mass = (true_dry_dep_denorm * cell_area).sum(dim=(-2, -1))
        pred_wet_dep_mass = (pred_wet_dep_denorm * cell_area).sum(dim=(-2, -1))
        true_wet_dep_mass = (true_wet_dep_denorm * cell_area).sum(dim=(-2, -1))

        # Total predicted and true mass
        pred_total_mass = pred_atm_mass + pred_dry_dep_mass + pred_wet_dep_mass
        true_total_mass = true_atm_mass + true_dry_dep_mass + true_wet_dep_mass

        # Source mass from metadata
        source_mass = metadata.get("emission_mass", metadata.get("source_mass", None))

        # Mass conservation loss
        if source_mass is not None:
            mass_conservation_loss = F.mse_loss(pred_total_mass, source_mass)
        else:
            # If no source mass is provided, assume mass should be conserved from true values
            mass_conservation_loss = F.mse_loss(pred_total_mass, true_total_mass)

        # Mass distribution loss (comparing distribution between prediction and truth)
        mass_distribution_loss = F.mse_loss(
            pred_total_mass / (pred_total_mass.sum() + 1e-8),
            true_total_mass / (true_total_mass.sum() + 1e-8),
        )

        return {
            "mass_conservation": mass_conservation_loss,
            "mass_distribution": mass_distribution_loss,
        }

    @staticmethod
    def calculate_mp_mass(
        mmr, ps, hyai, hybi, gw, P0, g=9.8
    ):  # g = 9.80665 but c'mon I'm an engineer
        """
        Calculate the total mass of microplastic particles across all altitude levels.

        Args:
            mmr: Microplastic mass mixing ratio [batch_size, n_particles, altitude, latitude, longitude]
            ps: Surface pressure [batch_size, latitude, longitude]
            hyai: Hybrid A coefficient for interfaces
            hybi: Hybrid B coefficient for interfaces
            gw: Gaussian weights for latitude
            P0: Reference pressure
            g: Gravitational acceleration (default: 9.8 m/s^2)

        Returns:
            mp_mass: Total mass of microplastic particles [batch_size, n_particles]
        """
        _, _, n_levels, _, _ = mmr.shape
        device = mmr.device

        if not isinstance(hyai, torch.Tensor):
            hyai = torch.tensor(hyai, dtype=torch.float32, device=device)

        if not isinstance(hybi, torch.Tensor):
            hybi = torch.tensor(hybi, dtype=torch.float32, device=device)

        if not isinstance(gw, torch.Tensor):
            gw = torch.tensor(gw, dtype=torch.float32, device=device)

        if not isinstance(P0, torch.Tensor):
            P0 = torch.tensor(P0, dtype=torch.float32, device=device)

        # (batch, 384, 576)
        ps = ps.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, 384, 576]

        hyai = hyai.view(1, 1, -1, 1, 1)  # [1, 1, 49, 1, 1]
        hybi = hybi.view(1, 1, -1, 1, 1)  # [1, 1, 49, 1, 1]

        # Pressure at interfaces
        # P_interface[k, i, j] = hyai[k] * P0 + hybi[k] * PS[i, j]
        p_interface = hyai * P0 + hybi * ps  # [batch, 1, 49, 384, 576]

        # Pressure differences
        # deltaP[k, i, j] = P_interface[k+1, i, j] - P_interface[k, i, j]
        delta_p = (
            p_interface[:, :, 1 : n_levels + 1] - p_interface[:, :, :n_levels]
        )  # [batch, 1, 48, 384, 576]

        # [1, 1, 1, 384, 1]
        gw = gw.view(1, 1, 1, -1, 1)

        # Compute air mass in each grid cell
        # air_mass[k, i, j] = (deltaP[k, i, j] * gw[i, j]) / g
        air_mass = (delta_p * gw) / g  # [batch, 1, 48, 384, 576]

        # Compute microplastic mass in each grid cell
        mp_cell_mass = mmr * air_mass  # [batch, n_particles, 48, 384, 576]

        mp_mass = mp_cell_mass.sum(dim=(2, 3, 4))  # [batch, n_particles]

        return mp_mass


class LinearModel(Base):
    def __init__(
        self,
        in_channels=11,
        out_channels=6,
        transport_loss_weight=1.0,
        use_physics_loss=False,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            transport_loss_weight=transport_loss_weight,
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
    x, mmr_true, metadata = next(iter(train_loader))

    # Create model instances - one with physics loss and one without
    model_no_physics = LinearModel(
        in_channels=11,
        out_channels=6,
        transport_loss_weight=1.0,
        use_physics_loss=False,
    )

    model_with_physics = LinearModel(
        in_channels=11,
        out_channels=6,
        transport_loss_weight=1.0,
        use_physics_loss=True,
    )

    dep_rand = torch.randn(1, 6, 2, 384, 576)

    # Test forward pass
    mmr_pred_no_physics = model_no_physics(x)
    mmr_pred_no_physics = torch.cat([mmr_pred_no_physics, dep_rand], dim=2)

    print(mmr_pred_no_physics.shape)

    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Output shape: {mmr_pred_no_physics.shape}")
    logger.info(f"Target shape: {mmr_true.shape}")

    # Compute losses for both models
    losses_no_physics = model_no_physics.compute_loss(x, mmr_pred_no_physics, mmr_true)
    logger.info("\nLosses without physics:")
    for name, value in losses_no_physics.items():
        logger.info(f"{name}: {value.item():.6f}")

    mmr_pred_physics = model_with_physics(x)
    mmr_pred_physics = torch.cat([mmr_pred_physics, dep_rand], dim=2)

    losses_physics = model_with_physics.compute_loss(x, mmr_pred_physics, mmr_true)
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
