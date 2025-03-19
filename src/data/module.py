import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional, Tuple, Dict, List
from src.utils import PATH, set_seed

set_seed()


class MPIDataset(Dataset):

    def __init__(
        self,
        h5_file: str = PATH.PROCESSED_DATA / "mpidata.h5",
        mode: str = "train",
        split: Tuple[float, float, float] = (0.5, 0.25, 0.25),
        normalize: bool = True,
        include_coordinates: bool = True,
    ):
        """
        Args:
            h5_file: Path to the HDF5 file
            mode: 'train', 'val', or 'test'
            split: Data split ratios (train, val, test)
            normalize: Whether to normalize the input and output data
            include_coordinates: Whether to include lat/lon coordinates as additional channels
        """
        super().__init__()

        self.h5_file = h5_file
        self.mode = mode
        self.split = split
        self.normalize = normalize
        self.include_coordinates = include_coordinates

        self._prepare_indices()

        # Metadata
        with h5py.File(self.h5_file, "r") as h5f:

            self.lats = h5f["metadata/lat"][:]
            self.lons = h5f["metadata/lon"][:]
            self.settling_velocities = h5f["metadata/settling_velocity"][:]

            ## Cell area
            lat_res = np.abs(self.lats[1] - self.lats[0])
            lon_res = np.abs(self.lons[1] - self.lons[0])

            # Earth radius
            R = 6371000

            # Mesh
            lat_grid, lon_grid = np.meshgrid(self.lats, self.lons, indexing="ij")

            self.cell_area = (
                np.radians(lat_res)
                * np.radians(lon_res)
                * R**2
                * np.cos(np.radians(lat_grid))
            )

            if self.normalize:
                self._compute_normalization_stats(h5f)

    def _prepare_indices(self):
        """
        In-case of shuffling
        """

        with h5py.File(self.h5_file, "r") as h5f:

            time_dim = h5f["metadata/time"].shape[0]

            all_indices = np.arange(time_dim)
            # np.random.shuffle(all_indices)

            train_size = int(time_dim * self.split[0])
            val_size = int(time_dim * self.split[1])

            if self.mode == "train":
                self.indices = all_indices[:train_size]
            elif self.mode == "val":
                self.indices = all_indices[train_size : train_size + val_size]
            elif self.mode == "test":
                self.indices = all_indices[train_size + val_size :]
            else:
                raise ValueError(f"Invalid mode: {self.mode}")

            self.num_samples = len(self.indices)

    def _compute_normalization_stats(self, h5f):

        # Avoid leakage
        train_indices = self.indices if self.mode == "train" else None

        if train_indices is None:

            time_dim = h5f["metadata/time"].shape[0]
            all_indices = np.arange(time_dim)

            np.random.shuffle(all_indices)

            train_size = int(time_dim * self.split[0])
            train_indices = all_indices[:train_size]

        self.input_mean = {}
        self.input_std = {}

        for var_name in h5f["inputs"].keys():
            data = h5f[f"inputs/{var_name}"][train_indices]

            self.input_mean[var_name] = float(np.mean(data))
            self.input_std[var_name] = float(np.std(data))

            if self.input_std[var_name] == 0:
                self.input_std[var_name] = 1.0

        self.output_mean = {"mmr": {}, "dry_dep": {}, "wet_dep": {}}
        self.output_std = {"mmr": {}, "dry_dep": {}, "wet_dep": {}}

        # TODO: DRY
        for size_idx in range(len(self.settling_velocities)):

            mmr_data = h5f[f"outputs/mass_mixing_ratio/size_{size_idx}"][train_indices]

            self.output_mean["mmr"][size_idx] = float(np.mean(mmr_data))
            self.output_std["mmr"][size_idx] = float(np.std(mmr_data))

            if self.output_std["mmr"][size_idx] == 0:
                self.output_std["mmr"][size_idx] = 1.0

            dry_dep_data = h5f[f"outputs/deposition/dry_size_{size_idx}"][train_indices]

            self.output_mean["dry_dep"][size_idx] = float(np.mean(dry_dep_data))
            self.output_std["dry_dep"][size_idx] = float(np.std(dry_dep_data))

            if self.output_std["dry_dep"][size_idx] == 0:
                self.output_std["dry_dep"][size_idx] = 1.0

            wet_dep_data = h5f[f"outputs/deposition/wet_size_{size_idx}"][train_indices]

            self.output_mean["wet_dep"][size_idx] = float(np.mean(wet_dep_data))
            self.output_std["wet_dep"][size_idx] = float(np.std(wet_dep_data))

            if self.output_std["wet_dep"][size_idx] == 0:
                self.output_std["wet_dep"][size_idx] = 1.0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        time_idx = self.indices[idx]

        with h5py.File(self.h5_file, "r") as h5f:
            # Order: PS, U, V, T, Q, TROPLEV, emissions
            ps = h5f["inputs/ps"][time_idx]
            u = h5f["inputs/u"][time_idx]
            v = h5f["inputs/v"][time_idx]
            t = h5f["inputs/t"][time_idx]
            q = h5f["inputs/q"][time_idx]
            troplev = h5f["inputs/troplev"][time_idx]
            emissions = h5f["inputs/emissions"][time_idx]

            if self.normalize:
                ps = (ps - self.input_mean["ps"]) / self.input_std["ps"]
                u = (u - self.input_mean["u"]) / self.input_std["u"]
                v = (v - self.input_mean["v"]) / self.input_std["v"]
                t = (t - self.input_mean["t"]) / self.input_std["t"]
                q = (q - self.input_mean["q"]) / self.input_std["q"]
                troplev = (troplev - self.input_mean["troplev"]) / self.input_std[
                    "troplev"
                ]
                emissions = (emissions - self.input_mean["emissions"]) / self.input_std[
                    "emissions"
                ]

            inputs = np.stack([ps, u, v, t, q, troplev, emissions], axis=0)

            if self.include_coordinates:
                # [-1, 1]
                lat_normalized = (
                    2
                    * (self.lats - self.lats.min())
                    / (self.lats.max() - self.lats.min())
                    - 1
                )

                lon_normalized = (
                    2
                    * (self.lons - self.lons.min())
                    / (self.lons.max() - self.lons.min())
                    - 1
                )

                # Create coordinate meshgrid
                lat_grid, lon_grid = np.meshgrid(
                    lat_normalized, lon_normalized, indexing="ij"
                )

                # Add coordinates as new channels
                inputs = np.concatenate(
                    [inputs, lat_grid[np.newaxis], lon_grid[np.newaxis]], axis=0
                )

            n_particles = len(self.settling_velocities)
            mmr = np.zeros((n_particles, len(self.lats), len(self.lons)))

            dry_dep = np.zeros_like(mmr)
            wet_dep = np.zeros_like(mmr)

            for size_idx in range(n_particles):
                mmr[size_idx] = h5f[f"outputs/mass_mixing_ratio/size_{size_idx}"][
                    time_idx
                ]
                dry_dep[size_idx] = h5f[f"outputs/deposition/dry_size_{size_idx}"][
                    time_idx
                ]
                wet_dep[size_idx] = h5f[f"outputs/deposition/wet_size_{size_idx}"][
                    time_idx
                ]

                if self.normalize:
                    mmr[size_idx] = (
                        mmr[size_idx] - self.output_mean["mmr"][size_idx]
                    ) / self.output_std["mmr"][size_idx]

                    dry_dep[size_idx] = (
                        dry_dep[size_idx] - self.output_mean["dry_dep"][size_idx]
                    ) / self.output_std["dry_dep"][size_idx]

                    wet_dep[size_idx] = (
                        wet_dep[size_idx] - self.output_mean["wet_dep"][size_idx]
                    ) / self.output_std["wet_dep"][size_idx]

        inputs = torch.tensor(inputs, dtype=torch.float32)

        mmr = torch.tensor(mmr, dtype=torch.float32)

        dry_dep = torch.tensor(dry_dep, dtype=torch.float32)
        wet_dep = torch.tensor(wet_dep, dtype=torch.float32)

        cell_area = torch.tensor(self.cell_area, dtype=torch.float32)

        return inputs, (mmr, dry_dep, wet_dep), cell_area


class MPIDataModule(pl.LightningDataModule):

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        normalize: bool = True,
        include_coordinates: bool = True,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split
        self.normalize = normalize
        self.include_coordinates = include_coordinates

        self.h5_file = PATH.PROCESSED_DATA / "mpidata.h5"

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        if not os.path.exists(self.h5_file):
            raise FileNotFoundError(
                f"HDF5 file not found at {self.h5_file}. "
                "Please run the data processing script first."
            )

    def setup(self, stage: Optional[str] = None):

        if stage == "fit" or stage is None:
            self.train_dataset = MPIDataset(
                h5_file=self.h5_file,
                mode="train",
                split=self.split,
                normalize=self.normalize,
                include_coordinates=self.include_coordinates,
            )

            self.val_dataset = MPIDataset(
                h5_file=self.h5_file,
                mode="val",
                split=self.split,
                normalize=self.normalize,
                include_coordinates=self.include_coordinates,
            )

        if stage == "test" or stage is None:
            self.test_dataset = MPIDataset(
                h5_file=self.h5_file,
                mode="test",
                split=self.split,
                normalize=self.normalize,
                include_coordinates=self.include_coordinates,
            )

    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_settling_velocities(self):

        with h5py.File(self.h5_file, "r") as h5f:
            return h5f["metadata/settling_velocity"][:]
