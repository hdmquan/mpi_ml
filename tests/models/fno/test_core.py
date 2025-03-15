import pytest
import torch
from src.models.fno import FNOModel


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def input_data(device):
    batch_size = 4
    height, width = 64, 64
    in_channels = 7

    return {
        "x": torch.randn(batch_size, in_channels, height, width, device=device),
        "batch_size": batch_size,
        "height": height,
        "width": width,
        "in_channels": in_channels,
        "out_channels": 6,
    }


@pytest.fixture
def fno_model(device, input_data):
    model = FNOModel(
        in_channels=input_data["in_channels"],
        out_channels=input_data["out_channels"],
        modes1=12,
        modes2=12,
        width=32,
        num_layers=2,
        include_coordinates=True,
    ).to(device)
    return model


@pytest.fixture
def fno_model_no_coords(device, input_data):
    model = FNOModel(
        in_channels=input_data["in_channels"],
        out_channels=input_data["out_channels"],
        include_coordinates=False,
    ).to(device)
    return model


def test_fno_with_coordinates(device, input_data, fno_model):
    x = input_data["x"]
    batch_size = input_data["batch_size"]
    height = input_data["height"]
    width = input_data["width"]
    out_channels = input_data["out_channels"]

    coords = fno_model.get_grid((height, width), device)
    coords = coords.repeat(batch_size, 1, 1, 1)

    assert coords.shape == (batch_size, 2, height, width)

    output = fno_model(x, coords)
    expected_shape = (batch_size, out_channels, height, width)

    assert output.shape == expected_shape

    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert (output >= 0).all(), "Output contains negative values"


def test_fno_without_coordinates(device, input_data, fno_model_no_coords):
    x = input_data["x"]
    batch_size = input_data["batch_size"]
    height = input_data["height"]
    width = input_data["width"]
    out_channels = input_data["out_channels"]

    output = fno_model_no_coords(x)
    expected_shape = (batch_size, out_channels, height, width)

    assert output.shape == expected_shape

    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert (output >= 0).all(), "Output contains negative values"
