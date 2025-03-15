import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import numpy as np
from pathlib import Path
import os

from src.utils import PATH


class MicroplasticDataset(Dataset):
    """
    Dataset for microplastic transport modeling.
    Loads data from the HDF5 file and preprocesses it for training.
    """

    def __init__(
        self,
        h5_file=None,
        mode="train",
        time_indices=None,
        level_indices=None,
        use_normalized=True,
        surface_level_only=False,
        include_derivatives=False,
        include_coordinates=True,
        input_variables=None,
        target_variables=None,
        transform=None,
    ):
        """
        Initialize the dataset

        Args:
            h5_file: Path to the HDF5 file
            mode: One of "train", "val", or "test"
            time_indices: List of time indices to use
            level_indices: List of vertical level indices to use
            use_normalized: Whether to use normalized variables
            surface_level_only: Whether to use only the surface level
            include_derivatives: Whether to include spatial derivatives
            include_coordinates: Whether to include lat/lon coordinates
            input_variables: List of input variable names
            target_variables: List of target variable names
            transform: Optional transform to apply to the data
        """
        super(MicroplasticDataset, self).__init__()

        # Set default file path if not provided
        if h5_file is None:
            h5_file = PATH.PROCESSED_DATA / "mpidata_v2.h5"

        self.h5_file = h5_file
        self.mode = mode
        self.use_normalized = use_normalized
        self.surface_level_only = surface_level_only
        self.include_derivatives = include_derivatives
        self.include_coordinates = include_coordinates
        self.transform = transform

        # Set default input variables if not provided
        if input_variables is None:
            self.input_variables = ["PS", "U", "V", "T", "Q", "TROPLEV"]
            # Add emission variables
            self.input_variables.extend(
                [f"Plast0{i+1}_SRF_EMIS_avrg" for i in range(6)]
            )
        else:
            self.input_variables = input_variables

        # Set default target variables if not provided
        if target_variables is None:
            self.target_variables = [f"Plast0{i+1}_MMR_avrg" for i in range(6)]
        else:
            self.target_variables = target_variables

        # Open the HDF5 file
        with h5py.File(self.h5_file, "r") as f:
            # Get dimensions
            self.num_times = f["coordinates/time"].shape[0]
            self.num_lats = f["coordinates/lat"].shape[0]
            self.num_lons = f["coordinates/lon"].shape[0]
            self.num_levels = f["coordinates/lev"].shape[0]

            # Get coordinates
            self.lats = f["coordinates/lat"][:]
            self.lons = f["coordinates/lon"][:]
            self.levels = f["coordinates/lev"][:]
            self.cell_area = f["coordinates/cell_area"][:]

            # Set time indices if not provided
            if time_indices is None:
                if mode == "train":
                    # Use first 70% of time steps for training
                    self.time_indices = list(range(int(self.num_times * 0.7)))
                elif mode == "val":
                    # Use next 15% of time steps for validation
                    self.time_indices = list(
                        range(
                            int(self.num_times * 0.7),
                            int(self.num_times * 0.85),
                        )
                    )
                else:  # mode == "test"
                    # Use last 15% of time steps for testing
                    self.time_indices = list(
                        range(
                            int(self.num_times * 0.85),
                            self.num_times,
                        )
                    )
            else:
                self.time_indices = time_indices

            # Set level indices if not provided
            if level_indices is None:
                if surface_level_only:
                    # Use only the surface level (index 0)
                    self.level_indices = [0]
                else:
                    # Use all levels
                    self.level_indices = list(range(self.num_levels))
            else:
                self.level_indices = level_indices

            # Store the number of samples
            self.num_samples = len(self.time_indices)

    def __len__(self):
        """
        Get the number of samples in the dataset

        Returns:
            Number of samples
        """
        return self.num_samples

    def __getitem__(self, idx):
        """
        Get a sample from the dataset

        Args:
            idx: Sample index

        Returns:
            Dictionary containing input data, ground truth, coordinates, etc.
        """
        # Get the time index
        time_idx = self.time_indices[idx]

        # Open the HDF5 file
        with h5py.File(self.h5_file, "r") as f:
            # Create normalized latitude and longitude grids
            lat_grid = (self.lats - self.lats.min()) / (
                self.lats.max() - self.lats.min()
            ) * 2 - 1
            lon_grid = (self.lons - self.lons.min()) / (
                self.lons.max() - self.lons.min()
            ) * 2 - 1

            # Create meshgrid
            lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

            # Create coordinates tensor
            coords = np.stack([lon_mesh, lat_mesh], axis=0)

            # Initialize input and target lists
            inputs = []
            targets = []

            # Load input variables
            for var in self.input_variables:
                if var in ["PS", "TROPLEV"] or var.endswith("SRF_EMIS_avrg"):
                    # 2D variables (time, lat, lon)
                    if self.use_normalized:
                        data = f[f"ml_features/normalized/{var}"][time_idx]
                    else:
                        if var.endswith("SRF_EMIS_avrg"):
                            data = f[f"boundary_conditions/surface_emissions/{var}"][
                                time_idx
                            ]
                        else:
                            data = f[f"physics/dynamics/{var}"][time_idx]

                    # Add a channel dimension
                    data = data[np.newaxis, :, :]
                else:
                    # 3D variables (time, lev, lat, lon)
                    if self.surface_level_only:
                        # Use only the surface level
                        if self.use_normalized:
                            data = f[f"ml_features/normalized/{var}"][time_idx, 0]
                        else:
                            if var.startswith("Plast"):
                                if int(var[5]) <= 4:
                                    data = f[
                                        f"physics/microplastics/small_particles/{var}"
                                    ][time_idx, 0]
                                else:
                                    data = f[
                                        f"physics/microplastics/large_particles/{var}"
                                    ][time_idx, 0]
                            else:
                                data = f[f"physics/dynamics/{var}"][time_idx, 0]

                        # Add a channel dimension
                        data = data[np.newaxis, :, :]
                    else:
                        # Use all specified levels
                        if self.use_normalized:
                            data = f[f"ml_features/normalized/{var}"][
                                time_idx, self.level_indices
                            ]
                        else:
                            if var.startswith("Plast"):
                                if int(var[5]) <= 4:
                                    data = f[
                                        f"physics/microplastics/small_particles/{var}"
                                    ][time_idx, self.level_indices]
                                else:
                                    data = f[
                                        f"physics/microplastics/large_particles/{var}"
                                    ][time_idx, self.level_indices]
                            else:
                                data = f[f"physics/dynamics/{var}"][
                                    time_idx, self.level_indices
                                ]

                # Append to inputs
                inputs.append(data)

            # Load target variables
            for var in self.target_variables:
                if self.surface_level_only:
                    # Use only the surface level
                    if self.use_normalized:
                        data = f[f"ml_features/normalized/{var}"][time_idx, 0]
                    else:
                        if int(var[5]) <= 4:
                            data = f[f"physics/microplastics/small_particles/{var}"][
                                time_idx, 0
                            ]
                        else:
                            data = f[f"physics/microplastics/large_particles/{var}"][
                                time_idx, 0
                            ]

                    # Add a channel dimension
                    data = data[np.newaxis, :, :]
                else:
                    # Use all specified levels
                    if self.use_normalized:
                        data = f[f"ml_features/normalized/{var}"][
                            time_idx, self.level_indices
                        ]
                    else:
                        if int(var[5]) <= 4:
                            data = f[f"physics/microplastics/small_particles/{var}"][
                                time_idx, self.level_indices
                            ]
                        else:
                            data = f[f"physics/microplastics/large_particles/{var}"][
                                time_idx, self.level_indices
                            ]

                # Append to targets
                targets.append(data)

            # Concatenate inputs and targets
            inputs = np.concatenate(inputs, axis=0)
            targets = np.concatenate(targets, axis=0)

            # Include derivatives if specified
            if self.include_derivatives and not self.surface_level_only:
                derivatives = []
                for var in self.target_variables:
                    # Load derivatives
                    dx = f[f"derivatives/{var}_dx"][time_idx, self.level_indices]
                    dy = f[f"derivatives/{var}_dy"][time_idx, self.level_indices]
                    dz = f[f"derivatives/{var}_dz"][time_idx, self.level_indices]

                    # Append to derivatives
                    derivatives.append(dx)
                    derivatives.append(dy)
                    derivatives.append(dz)

                # Concatenate derivatives
                derivatives = np.concatenate(derivatives, axis=0)

                # Convert to torch tensors
                inputs = torch.from_numpy(inputs).float()
                targets = torch.from_numpy(targets).float()
                derivatives = torch.from_numpy(derivatives).float()
                coords = torch.from_numpy(coords).float()
                cell_area = torch.from_numpy(self.cell_area).float()

                # Apply transform if specified
                if self.transform is not None:
                    inputs, targets, derivatives, coords = self.transform(
                        inputs, targets, derivatives, coords
                    )

                # Return dictionary
                return {
                    "inputs": inputs,
                    "targets": targets,
                    "derivatives": derivatives,
                    "coords": coords,
                    "cell_area": cell_area,
                    "time_idx": time_idx,
                }
            else:
                # Convert to torch tensors
                inputs = torch.from_numpy(inputs).float()
                targets = torch.from_numpy(targets).float()
                coords = torch.from_numpy(coords).float()
                cell_area = torch.from_numpy(self.cell_area).float()

                # Apply transform if specified
                if self.transform is not None:
                    inputs, targets, coords = self.transform(inputs, targets, coords)

                # Return dictionary
                return {
                    "inputs": inputs,
                    "targets": targets,
                    "coords": coords,
                    "cell_area": cell_area,
                    "time_idx": time_idx,
                }


class MicroplasticDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for microplastic transport modeling.
    """

    def __init__(
        self,
        h5_file=None,
        batch_size=4,
        num_workers=4,
        use_normalized=True,
        surface_level_only=True,
        include_derivatives=False,
        include_coordinates=True,
        input_variables=None,
        target_variables=None,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_split_seed=42,
    ):
        """
        Initialize the data module

        Args:
            h5_file: Path to the HDF5 file
            batch_size: Batch size
            num_workers: Number of workers for data loading
            use_normalized: Whether to use normalized variables
            surface_level_only: Whether to use only the surface level
            include_derivatives: Whether to include spatial derivatives
            include_coordinates: Whether to include lat/lon coordinates
            input_variables: List of input variable names
            target_variables: List of target variable names
            train_ratio: Ratio of data to use for training
            val_ratio: Ratio of data to use for validation
            test_ratio: Ratio of data to use for testing
            random_split_seed: Random seed for splitting the data
        """
        super(MicroplasticDataModule, self).__init__()

        # Set default file path if not provided
        if h5_file is None:
            h5_file = PATH.PROCESSED_DATA / "mpidata_v2.h5"

        self.h5_file = h5_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_normalized = use_normalized
        self.surface_level_only = surface_level_only
        self.include_derivatives = include_derivatives
        self.include_coordinates = include_coordinates
        self.input_variables = input_variables
        self.target_variables = target_variables
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_split_seed = random_split_seed

        # Validate ratios
        assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

    def prepare_data(self):
        """
        Prepare the data (download, etc.)
        """
        # Check if the HDF5 file exists
        if not os.path.exists(self.h5_file):
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_file}")

    def setup(self, stage=None):
        """
        Set up the data module

        Args:
            stage: Stage ("fit", "validate", "test", or "predict")
        """
        # Open the HDF5 file to get the number of time steps
        with h5py.File(self.h5_file, "r") as f:
            num_times = f["coordinates/time"].shape[0]

        # Create a list of all time indices
        all_indices = list(range(num_times))

        # Set the random seed for reproducibility
        torch.manual_seed(self.random_split_seed)

        # Split the indices into train, val, and test sets
        train_size = int(num_times * self.train_ratio)
        val_size = int(num_times * self.val_ratio)
        test_size = num_times - train_size - val_size

        # Randomly split the indices
        train_indices, val_indices, test_indices = random_split(
            all_indices, [train_size, val_size, test_size]
        )

        # Convert to lists
        train_indices = list(train_indices)
        val_indices = list(val_indices)
        test_indices = list(test_indices)

        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = MicroplasticDataset(
                h5_file=self.h5_file,
                mode="train",
                time_indices=train_indices,
                use_normalized=self.use_normalized,
                surface_level_only=self.surface_level_only,
                include_derivatives=self.include_derivatives,
                include_coordinates=self.include_coordinates,
                input_variables=self.input_variables,
                target_variables=self.target_variables,
            )

            self.val_dataset = MicroplasticDataset(
                h5_file=self.h5_file,
                mode="val",
                time_indices=val_indices,
                use_normalized=self.use_normalized,
                surface_level_only=self.surface_level_only,
                include_derivatives=self.include_derivatives,
                include_coordinates=self.include_coordinates,
                input_variables=self.input_variables,
                target_variables=self.target_variables,
            )

        if stage == "test" or stage is None:
            self.test_dataset = MicroplasticDataset(
                h5_file=self.h5_file,
                mode="test",
                time_indices=test_indices,
                use_normalized=self.use_normalized,
                surface_level_only=self.surface_level_only,
                include_derivatives=self.include_derivatives,
                include_coordinates=self.include_coordinates,
                input_variables=self.input_variables,
                target_variables=self.target_variables,
            )

    def train_dataloader(self):
        """
        Get the training data loader

        Returns:
            Training data loader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """
        Get the validation data loader

        Returns:
            Validation data loader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        """
        Get the test data loader

        Returns:
            Test data loader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
