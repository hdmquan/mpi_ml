import pytest
import torch
from src.models.fno import FNOModel2D, FNOModel3D


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def input_data_2d(device):
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
def input_data_3d(device):
    batch_size = 4
    height, width, levels = 64, 64, 64
    in_channels = 7

    return {
        "x": torch.randn(batch_size, in_channels, levels, height, width, device=device),
        "batch_size": batch_size,
        "height": height,
        "width": width,
        "levels": levels,
        "in_channels": in_channels,
        "out_channels": 6,
    }


@pytest.fixture
def fno_model_2d_no_coords(device, input_data_2d):
    model = FNOModel2D(
        in_channels=input_data_2d["in_channels"],
        out_channels=input_data_2d["out_channels"],
        include_coordinates=False,
    ).to(device)
    return model


@pytest.fixture
def fno_model_3d(device, input_data_3d):
    model = FNOModel3D(
        in_channels=input_data_3d["in_channels"],
        out_channels=input_data_3d["out_channels"],
        modes1=12,
        modes2=12,
        modes3=12,
        width=32,
        num_layers=2,
        include_coordinates=True,
    ).to(device)
    return model


@pytest.fixture
def fno_model_3d_no_coords(device, input_data_3d):
    model = FNOModel3D(
        in_channels=input_data_3d["in_channels"],
        out_channels=input_data_3d["out_channels"],
        include_coordinates=False,
    ).to(device)
    return model


def test_fno_2d_with_coordinates(device, input_data_2d, fno_model_2d):
    x = input_data_2d["x"]
    batch_size = input_data_2d["batch_size"]
    height = input_data_2d["height"]
    width = input_data_2d["width"]
    out_channels = input_data_2d["out_channels"]

    coords = fno_model_2d.get_grid((height, width), device)
    coords = coords.repeat(batch_size, 1, 1, 1)

    assert coords.shape == (batch_size, 2, height, width)

    output = fno_model_2d(x, coords)
    expected_shape = (batch_size, out_channels, height, width)

    assert output.shape == expected_shape

    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert (output >= 0).all(), "Output contains negative values"


def test_fno_2d_without_coordinates(device, input_data_2d, fno_model_2d_no_coords):
    x = input_data_2d["x"]
    batch_size = input_data_2d["batch_size"]
    height = input_data_2d["height"]
    width = input_data_2d["width"]
    out_channels = input_data_2d["out_channels"]

    output = fno_model_2d_no_coords(x)
    expected_shape = (batch_size, out_channels, height, width)

    assert output.shape == expected_shape

    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert (output >= 0).all(), "Output contains negative values"


def test_fno_3d_with_coordinates(device, input_data_3d, fno_model_3d):
    x = input_data_3d["x"]
    batch_size = input_data_3d["batch_size"]
    height = input_data_3d["height"]
    width = input_data_3d["width"]
    levels = input_data_3d["levels"]
    out_channels = input_data_3d["out_channels"]

    coords = fno_model_3d.get_grid((levels, height, width), device)
    coords = coords.repeat(batch_size, 1, 1, 1, 1)

    assert coords.shape == (batch_size, 3, levels, height, width)

    output = fno_model_3d(x, coords)
    expected_shape = (batch_size, out_channels, levels, height, width)

    assert output.shape == expected_shape

    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert (output >= 0).all(), "Output contains negative values"


def test_fno_3d_without_coordinates(device, input_data_3d, fno_model_3d_no_coords):
    x = input_data_3d["x"]
    batch_size = input_data_3d["batch_size"]
    height = input_data_3d["height"]
    width = input_data_3d["width"]
    levels = input_data_3d["levels"]
    out_channels = input_data_3d["out_channels"]

    output = fno_model_3d_no_coords(x)
    expected_shape = (batch_size, out_channels, levels, height, width)

    assert output.shape == expected_shape

    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert (output >= 0).all(), "Output contains negative values"
