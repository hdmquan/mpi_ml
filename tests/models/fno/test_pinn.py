import pytest
import torch
from src.models.fno.pinn import PINNModel
from src.utils import set_seed

set_seed()


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def model_params():
    return {
        "in_channels": 7,
        "out_channels": 6,
        "modes1": 8,
        "modes2": 8,
        "width": 32,
        "num_layers": 2,
        "physics_weight": 0.1,
        "boundary_weight": 0.1,
        "conservation_weight": 0.1,
        "include_coordinates": True,
    }


@pytest.fixture
def test_data(device):
    batch_size = 2
    height, width = 32, 64
    in_channels = 7
    out_channels = 6

    x = torch.randn(batch_size, in_channels, height, width, device=device)

    x[:, -out_channels:] = torch.abs(
        torch.randn(batch_size, out_channels, height, width, device=device)
    )

    y_true = torch.abs(
        torch.randn(batch_size, out_channels, height, width, device=device)
    )

    lat = torch.linspace(-90, 90, height, device=device)
    lon = torch.linspace(0, 360, width, device=device)
    lon_grid, lat_grid = torch.meshgrid(lon, lat, indexing="xy")
    coords = torch.stack([lon_grid, lat_grid], dim=0)
    coords = coords.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    cell_area = torch.cos(torch.deg2rad(lat_grid)).unsqueeze(0).repeat(batch_size, 1, 1)

    return {
        "x": x,
        "y_true": y_true,
        "coords": coords,
        "cell_area": cell_area,
        "batch_size": batch_size,
        "height": height,
        "width": width,
        "in_channels": in_channels,
        "out_channels": out_channels,
    }


@pytest.fixture
def pinn_model(device, model_params):
    model = PINNModel(**model_params).to(device)
    return model


def test_model_initialization(pinn_model, model_params):
    """Test that the model initializes with the correct parameters."""
    for key, value in model_params.items():
        assert (
            pinn_model.hparams[key] == value
        ), f"Expected {key}={value}, got {pinn_model.hparams[key]}"

    assert isinstance(
        pinn_model.fno_model, torch.nn.Module
    ), "FNO model not properly initialized"
    assert hasattr(pinn_model, "settling_vel"), "Settling velocities not initialized"


def test_forward_pass(pinn_model, test_data, device):
    """Test the forward pass of the model."""
    x = test_data["x"]
    coords = test_data["coords"]

    output = pinn_model(x, coords)

    expected_shape = (
        test_data["batch_size"],
        test_data["out_channels"],
        test_data["height"],
        test_data["width"],
    )

    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert (
        output >= 0
    ).all(), "Output contains negative values (expected non-negative from activation)"


def test_physics_loss_computation(pinn_model, test_data):
    """Test the physics-informed loss computation."""
    x = test_data["x"]
    y_true = test_data["y_true"]
    coords = test_data["coords"]
    cell_area = test_data["cell_area"]

    y_pred = pinn_model(x, coords)

    physics_losses = pinn_model.compute_physics_loss(x, y_pred, y_true, cell_area)

    assert "conservation_loss" in physics_losses, "Conservation loss missing"
    assert "boundary_loss" in physics_losses, "Boundary loss missing"
    assert "mass_conservation_loss" in physics_losses, "Mass conservation loss missing"

    for loss_name, loss_value in physics_losses.items():
        assert loss_value >= 0, f"{loss_name} should be non-negative"
        assert torch.isfinite(loss_value), f"{loss_name} should be finite"


def test_nn_loss_computation(pinn_model, test_data):
    """Test the neural network loss computation."""
    x = test_data["x"]
    y_true = test_data["y_true"]
    coords = test_data["coords"]
    cell_area = test_data["cell_area"]

    y_pred = pinn_model(x, coords)

    physics_losses, nn_loss, total_loss = pinn_model.compute_nn_loss(
        x, y_pred, y_true, cell_area
    )

    assert nn_loss >= 0, "NN loss should be non-negative"
    assert torch.isfinite(nn_loss), "NN loss should be finite"
    assert total_loss >= 0, "Total loss should be non-negative"
    assert torch.isfinite(total_loss), "Total loss should be finite"

    expected_total = (
        nn_loss
        + pinn_model.hparams.physics_weight * physics_losses["conservation_loss"]
        + pinn_model.hparams.boundary_weight * physics_losses["boundary_loss"]
        + pinn_model.hparams.conservation_weight
        * physics_losses["mass_conservation_loss"]
    )

    assert torch.isclose(
        total_loss, expected_total
    ), "Total loss calculation is incorrect"


def test_training_step(pinn_model, test_data):
    """Test the training step."""
    batch = {
        "inputs": test_data["x"],
        "targets": test_data["y_true"],
        "coords": test_data["coords"],
        "cell_area": test_data["cell_area"],
    }

    # Mock the trainer or suppress logging
    # Option 1: Temporarily disable logging
    original_log = pinn_model.log
    pinn_model.log = lambda *args, **kwargs: None

    try:
        loss = pinn_model.training_step(batch, 0)
    finally:
        # Restore original log method
        pinn_model.log = original_log

    assert torch.is_tensor(loss), "Training step should return a tensor"
    assert loss.dim() == 0, "Loss should be a scalar tensor"
    assert loss >= 0, "Loss should be non-negative"
    assert torch.isfinite(loss), "Loss should be finite"


def test_validation_step(pinn_model, test_data):
    """Test the validation step."""
    batch = {
        "inputs": test_data["x"],
        "targets": test_data["y_true"],
        "coords": test_data["coords"],
        "cell_area": test_data["cell_area"],
    }

    # Temporarily disable logging
    original_log = pinn_model.log
    pinn_model.log = lambda *args, **kwargs: None

    try:
        loss = pinn_model.validation_step(batch, 0)
    finally:
        # Restore original log method
        pinn_model.log = original_log

    assert torch.is_tensor(loss), "Validation step should return a tensor"
    assert loss.dim() == 0, "Loss should be a scalar tensor"
    assert loss >= 0, "Loss should be non-negative"
    assert torch.isfinite(loss), "Loss should be finite"


def test_test_step(pinn_model, test_data):
    """Test the test step."""
    batch = {
        "inputs": test_data["x"],
        "targets": test_data["y_true"],
        "coords": test_data["coords"],
        "cell_area": test_data["cell_area"],
    }

    # Temporarily disable logging
    original_log = pinn_model.log
    pinn_model.log = lambda *args, **kwargs: None

    try:
        loss = pinn_model.test_step(batch, 0)
    finally:
        # Restore original log method
        pinn_model.log = original_log

    assert torch.is_tensor(loss), "Test step should return a tensor"
    assert loss.dim() == 0, "Loss should be a scalar tensor"
    assert loss >= 0, "Loss should be non-negative"
    assert torch.isfinite(loss), "Loss should be finite"


def test_optimizer_configuration(pinn_model):
    """Test the optimizer configuration."""
    optimizer_config = pinn_model.configure_optimizers()

    assert "optimizer" in optimizer_config, "Optimizer missing from configuration"
    assert (
        "lr_scheduler" in optimizer_config
    ), "Learning rate scheduler missing from configuration"
    assert "monitor" in optimizer_config, "Monitor metric missing from configuration"

    assert isinstance(
        optimizer_config["optimizer"], torch.optim.Optimizer
    ), "Optimizer should be a PyTorch optimizer"

    assert hasattr(
        optimizer_config["lr_scheduler"], "step"
    ), "LR scheduler should have a 'step' method"

    assert (
        optimizer_config["monitor"] == "val_total_loss"
    ), "Monitor metric should be 'val_total_loss'"
