import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional, Tuple, Dict, List
from src.utils import PATH, set_seed
from loguru import logger

set_seed()

# Quantization ◁ △ ▷ ▽ ◁ △ ▷ ▽
DTYPE = np.float32


class MPIDataset(Dataset):

    def __init__(
        self,
        h5_file: str = PATH.PROCESSED_DATA / "mpidata.h5",
        mode: str = "train",
        split: Tuple[float, float, float] = (0.5, 0.25, 0.25),
        normalize: bool = True,
        include_coordinates: bool = True,
        shuffle: bool = False,
        reverse_lev: bool = True,
        include_troplev: bool = False,
        only_mmr: bool = True,
    ):
        """
        Args:
            h5_file: Path to the HDF5 file
            mode: 'train', 'val', or 'test'
            split: Data split ratios (train, val, test)
            normalize: Whether to normalize the input and output data
            include_coordinates: Whether to include lat/lon coordinates as additional channels
            shuffle: Whether to shuffle the data indices (default: False)
            reverse_lev: Latitude originally descending (default: True)
        """
        super().__init__()

        self.h5_file = h5_file
        self.mode = mode
        self.split = split
        self.normalize = normalize
        self.include_coordinates = include_coordinates
        self.shuffle = shuffle
        self.reverse_lev = reverse_lev
        self.include_troplev = include_troplev
        self.only_mmr = only_mmr
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

        # Hardcoded MMR normalization values
        self.mmr_min = -0.8929
        self.mmr_max = 500.0

    def _prepare_indices(self):
        """
        For data splitting
        """
        with h5py.File(self.h5_file, "r") as h5f:
            time_dim = h5f["metadata/time"].shape[0]
            all_indices = np.arange(time_dim)

            if self.shuffle:
                rng = np.random.default_rng()
                shuffled_indices = rng.permutation(all_indices)
            else:
                shuffled_indices = all_indices

            train_size = int(time_dim * self.split[0])
            val_size = int(time_dim * self.split[1])

            if self.mode == "train":
                self.indices = np.sort(shuffled_indices[:train_size])
            elif self.mode == "val":
                self.indices = np.sort(
                    shuffled_indices[train_size : train_size + val_size]
                )
            elif self.mode == "test":
                self.indices = np.sort(shuffled_indices[train_size + val_size :])
            else:
                raise ValueError(f"Invalid mode: {self.mode}")

            self.num_samples = len(self.indices)

    def _compute_normalization_stats(self, h5f):
        # Avoid leakage
        train_indices = self.indices if self.mode == "train" else None

        if train_indices is None:
            time_dim = h5f["metadata/time"].shape[0]
            if self.shuffle:
                shuffled_indices = np.random.permutation(time_dim)
            else:
                shuffled_indices = np.arange(time_dim)

            train_size = int(time_dim * self.split[0])
            train_indices = np.sort(shuffled_indices[:train_size])

        self.input_mean = {}
        self.input_std = {}

        met_vars = ["PS", "U", "V", "T", "Q"]

        if self.include_troplev:
            met_vars.append("TROPLEV")

        for var_name in met_vars:
            data = h5f[f"inputs/{var_name}"][train_indices]
            self.input_mean[var_name] = float(np.mean(data))
            self.input_std[var_name] = float(np.std(data))
            if self.input_std[var_name] == 0:
                self.input_std[var_name] = 1.0

        self.output_mean = {"mmr": {}, "dry_dep": {}, "wet_dep": {}}
        self.output_std = {"mmr": {}, "dry_dep": {}, "wet_dep": {}}

        for size_idx in range(len(self.settling_velocities)):
            # Update paths to match process.py structure
            mmr_data = h5f[f"outputs/mass_mixing_ratio/Plast0{size_idx+1}_MMR_avrg"][
                train_indices
            ]

            self.output_mean["mmr"][size_idx] = float(np.mean(mmr_data))
            self.output_std["mmr"][size_idx] = float(np.std(mmr_data))
            if self.output_std["mmr"][size_idx] == 0:
                self.output_std["mmr"][size_idx] = 1.0

            if not self.only_mmr:  # Only compute deposition stats if we're using them
                dry_dep_data = h5f[
                    f"outputs/deposition/Plast0{size_idx+1}_DRY_DEP_FLX_avrg"
                ][train_indices]
                wet_dep_data = h5f[
                    f"outputs/deposition/Plast0{size_idx+1}_WETDEP_FLUX_avrg"
                ][train_indices]

                self.output_mean["dry_dep"][size_idx] = float(np.mean(dry_dep_data))
                self.output_std["dry_dep"][size_idx] = float(np.std(dry_dep_data))
                if self.output_std["dry_dep"][size_idx] == 0:
                    self.output_std["dry_dep"][size_idx] = 1.0

                self.output_mean["wet_dep"][size_idx] = float(np.mean(wet_dep_data))
                self.output_std["wet_dep"][size_idx] = float(np.std(wet_dep_data))
                if self.output_std["wet_dep"][size_idx] == 0:
                    self.output_std["wet_dep"][size_idx] = 1.0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        time_idx = self.indices[idx]

        with h5py.File(self.h5_file, "r") as h5f:
            # Get 2D fields - slice directly instead of loading entire arrays
            ps = h5f["inputs/PS"][time_idx].astype(DTYPE)
            if self.include_troplev:
                troplev = h5f["inputs/TROPLEV"][time_idx].astype(DTYPE)

            # Get 3D fields - slice directly and convert to float32
            u = h5f["inputs/U"][time_idx].astype(DTYPE)
            v = h5f["inputs/V"][time_idx].astype(DTYPE)
            t = h5f["inputs/T"][time_idx].astype(DTYPE)
            q = h5f["inputs/Q"][time_idx].astype(DTYPE)

            if self.normalize:
                ps = (ps - self.input_mean["PS"]) / self.input_std["PS"]
                if self.include_troplev:
                    troplev = (troplev - self.input_mean["TROPLEV"]) / self.input_std[
                        "TROPLEV"
                    ]
                u = (u - self.input_mean["U"]) / self.input_std["U"]
                v = (v - self.input_mean["V"]) / self.input_std["V"]
                t = (t - self.input_mean["T"]) / self.input_std["T"]
                q = (q - self.input_mean["Q"]) / self.input_std["Q"]

            # Create sparse coordinates once during initialization instead of for each item
            if not hasattr(self, "lat_grid_3d") and self.include_coordinates:
                self._create_coordinate_grids()

            # Repeat the 2D fields to match the 3D field shape
            shape = u.shape
            ps_3d = np.reshape(ps, (1,) + ps.shape).repeat(shape[0], axis=0)
            if self.include_troplev:
                troplev_3d = np.reshape(troplev, (1,) + troplev.shape).repeat(
                    shape[0], axis=0
                )

            # Stack inputs with reduced memory footprint
            inputs = np.stack([ps_3d, u, v, t, q], axis=0)
            if self.include_troplev:
                inputs = np.concatenate([inputs, troplev_3d], axis=0)

            if self.include_coordinates:
                inputs = np.concatenate(
                    [inputs, self.lat_grid_3d, self.lon_grid_3d], axis=0
                )

            n_particles = len(self.settling_velocities)
            n_levels = shape[0]  # Number of vertical levels

            # Initialize with float32 for smaller memory footprint
            mmr = np.zeros(
                (n_particles, n_levels, len(self.lats), len(self.lons)),
                dtype=DTYPE,
            )

            for size_idx in range(n_particles):
                mmr[size_idx] = h5f[
                    f"outputs/mass_mixing_ratio/Plast0{size_idx+1}_MMR_avrg"
                ][time_idx].astype(DTYPE)

                if self.normalize:
                    # Use min-max normalization instead of mean-std
                    mmr[size_idx] = (mmr[size_idx] - self.mmr_min) / (
                        self.mmr_max - self.mmr_min
                    )

            # Convert to tensors
            inputs = torch.from_numpy(inputs)
            mmr = torch.from_numpy(mmr)

            if self.only_mmr:
                return inputs, mmr
            else:
                dry_dep = np.zeros(
                    (n_particles, len(self.lats), len(self.lons)), dtype=DTYPE
                )
                wet_dep = np.zeros(
                    (n_particles, len(self.lats), len(self.lons)), dtype=DTYPE
                )

                for size_idx in range(n_particles):
                    dry_dep[size_idx] = h5f[
                        f"outputs/deposition/Plast0{size_idx+1}_DRY_DEP_FLX_avrg"
                    ][time_idx].astype(DTYPE)
                    wet_dep[size_idx] = h5f[
                        f"outputs/deposition/Plast0{size_idx+1}_WETDEP_FLUX_avrg"
                    ][time_idx].astype(DTYPE)

                    if self.normalize:
                        dry_dep[size_idx] = (
                            dry_dep[size_idx] - self.output_mean["dry_dep"][size_idx]
                        ) / self.output_std["dry_dep"][size_idx]
                        wet_dep[size_idx] = (
                            wet_dep[size_idx] - self.output_mean["wet_dep"][size_idx]
                        ) / self.output_std["wet_dep"][size_idx]

                dry_dep = torch.from_numpy(dry_dep)
                wet_dep = torch.from_numpy(wet_dep)
                return inputs, (mmr, dry_dep, wet_dep)

    def _create_coordinate_grids(self):
        """Pre-compute coordinate grids once to save memory and computation"""
        # [-1, 1] normalized coordinates
        lat_normalized = (
            2 * (self.lats - self.lats.min()) / (self.lats.max() - self.lats.min()) - 1
        )
        lon_normalized = (
            2 * (self.lons - self.lons.min()) / (self.lons.max() - self.lons.min()) - 1
        )

        # Create coordinate meshgrid
        lat_grid, lon_grid = np.meshgrid(lat_normalized, lon_normalized, indexing="ij")

        # Create 3D coordinate grids
        with h5py.File(self.h5_file, "r") as h5f:
            # Sample one 3D field to get dimensions
            sample_3d = h5f["inputs/U"][0]
            n_levels = sample_3d.shape[0]

        # Store as class attributes in the right shape for stacking
        self.lat_grid_3d = np.broadcast_to(
            lat_grid[np.newaxis, :, :],
            (1, n_levels, lat_grid.shape[0], lat_grid.shape[1]),
        ).astype(DTYPE)
        self.lon_grid_3d = np.broadcast_to(
            lon_grid[np.newaxis, :, :],
            (1, n_levels, lon_grid.shape[0], lon_grid.shape[1]),
        ).astype(DTYPE)


class MPIDataModule(pl.LightningDataModule):

    def __init__(
        self,
        batch_size: int = 1,
        num_workers: int = 4,
        split: Tuple[float, float, float] = (0.5, 0.25, 0.25),
        normalize: bool = True,
        include_coordinates: bool = True,
        **kwargs,
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
                shuffle=True,
            )

            self.val_dataset = MPIDataset(
                h5_file=self.h5_file,
                mode="val",
                split=self.split,
                normalize=self.normalize,
                include_coordinates=self.include_coordinates,
                shuffle=True,
            )

        if stage == "test" or stage is None:
            self.test_dataset = MPIDataset(
                h5_file=self.h5_file,
                mode="test",
                split=self.split,
                normalize=self.normalize,
                include_coordinates=self.include_coordinates,
                shuffle=True,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def get_settling_velocities(self):

        with h5py.File(self.h5_file, "r") as h5f:
            return h5f["metadata/settling_velocity"][:]


if __name__ == "__main__":
    from src.utils.memory import print_memory_usage, print_tensor_memory

    print_memory_usage()

    datamodule = MPIDataModule()
    datamodule.setup()

    print_memory_usage()

    train_dataloader = datamodule.train_dataloader()

    print(len(train_dataloader))

    for batch in train_dataloader:
        print(batch[0].shape)  # [1, 8, 48, 384, 576]
        print_tensor_memory(batch[0])
        print(batch[1].shape)  # [1, 6, 48, 384, 576]
        print_tensor_memory(batch[1])
        # print(batch[1][0].shape)  # [1, 6, 48, 384, 576]
        # print(batch[1][1].shape)  # [1, 6, 384, 576]
        # print(batch[1][2].shape)  # [1, 6, 384, 576]
        print(batch[2].shape)  # [1, 384, 576]
        print_tensor_memory(batch[2])
        print_memory_usage()

        break
