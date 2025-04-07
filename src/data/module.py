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
        force_normalize: bool = False,
        include_emissions: bool = True,
        include_metadata: bool = True,
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
            include_emissions: Whether to include surface emissions as inputs (default: True)
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
        self.force_normalize = force_normalize
        self.include_emissions = include_emissions
        self.include_metadata = include_metadata
        self._prepare_indices()
        self._file_handle = None

        # Metadata
        with h5py.File(self.h5_file, "r") as h5f:
            self.lats = h5f["metadata/lat"][:]
            self.lons = h5f["metadata/lon"][:]
            self.settling_velocities = h5f["metadata/settling_velocity"][:]

            self.P0 = h5f["metadata/P0"][()]
            self.gw = h5f["metadata/gw"][()]
            self.hyai = h5f["metadata/hyai"][()]
            self.hybi = h5f["metadata/hybi"][()]

            R_earth = 6371000

            self.cell_area = 2 * np.pi * R_earth**2 * self.gw[:, np.newaxis]
            self._normalize(h5f, force_normalize=force_normalize)

            raw_emission = []
            emission_tensors = []
            if self.include_emissions:
                n_particles = len(self.settling_velocities)
                for size_idx in range(n_particles):
                    # Get emission data
                    emission_raw = h5f[
                        f"outputs/emissions/Plast0{size_idx+1}_SRF_EMIS_avrg"
                    ][0]
                    emission_tensor = torch.from_numpy(emission_raw).to(torch.float32)

                    raw_emission.append(emission_raw)
                    # Normalize if needed
                    if self.normalize:
                        emission_tensor = (
                            emission_tensor - self.emission_mean[size_idx]
                        ) / self.emission_std[size_idx]

                    # TODO: Hardcoded for now
                    emission_3d = emission_tensor.unsqueeze(0).repeat(48, 1, 1)
                    emission_tensors.append(emission_3d)

                self.emissions = torch.stack(emission_tensors)

                raw_emission = np.stack(raw_emission)
                emission_mass = raw_emission.sum(axis=0) * self.cell_area
                self.emission_mass = torch.sum(
                    torch.from_numpy(emission_mass).to(torch.float32)
                )

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

    def _normalize(self, h5f, force_normalize: bool = False):
        """Compute and cache normalization statistics for the dataset

        Args:
            h5f: Open h5py file handle
            force_normalize: If True, recompute normalization even if cached
        """
        # Check if normalization exists and we're not forcing recomputation
        if not force_normalize and "normalization" in h5f:
            print("Loading existing normalization stats")
            # Load existing normalization stats
            norm_group = h5f["normalization"]

            # Load input stats
            self.input_mean = {}
            self.input_std = {}
            met_vars = ["PS", "U", "V", "T", "Q"]
            if self.include_troplev:
                met_vars.append("TROPLEV")

            for var in met_vars:
                if f"{var}_mean" in norm_group and f"{var}_std" in norm_group:
                    self.input_mean[var] = float(norm_group[f"{var}_mean"][()])
                    self.input_std[var] = float(norm_group[f"{var}_std"][()])
                else:
                    return False  # Missing some stats, need to recompute

            # Load emission stats if needed
            if self.include_emissions:
                self.emission_mean = {}
                self.emission_std = {}
                n_particles = len(self.settling_velocities)

                for size_idx in range(n_particles):
                    stat_key = f"emission_{size_idx}"
                    if (
                        f"{stat_key}_mean" in norm_group
                        and f"{stat_key}_std" in norm_group
                    ):
                        self.emission_mean[size_idx] = float(
                            norm_group[f"{stat_key}_mean"][()]
                        )
                        self.emission_std[size_idx] = float(
                            norm_group[f"{stat_key}_std"][()]
                        )
                    else:
                        return False  # Missing emission stats, need to recompute

            # Load output stats
            self.output_mean = {"mmr": {}, "dry_dep": {}, "wet_dep": {}}
            self.output_std = {"mmr": {}, "dry_dep": {}, "wet_dep": {}}

            n_particles = len(self.settling_velocities)
            for size_idx in range(n_particles):
                # Load MMR stats
                if (
                    f"mmr_{size_idx}_mean" in norm_group
                    and f"mmr_{size_idx}_std" in norm_group
                ):
                    self.output_mean["mmr"][size_idx] = float(
                        norm_group[f"mmr_{size_idx}_mean"][()]
                    )
                    self.output_std["mmr"][size_idx] = float(
                        norm_group[f"mmr_{size_idx}_std"][()]
                    )
                else:
                    return False

                if not self.only_mmr:
                    # Load deposition stats
                    for dep_type in ["dry_dep", "wet_dep"]:
                        if (
                            f"{dep_type}_{size_idx}_mean" in norm_group
                            and f"{dep_type}_{size_idx}_std" in norm_group
                        ):
                            self.output_mean[dep_type][size_idx] = float(
                                norm_group[f"{dep_type}_{size_idx}_mean"][()]
                            )
                            self.output_std[dep_type][size_idx] = float(
                                norm_group[f"{dep_type}_{size_idx}_std"][()]
                            )
                        else:
                            return False

            # Load MMR log-scale stats
            if (
                "mmr_min" in norm_group
                and "mmr_max" in norm_group
                and "mmr_epsilon" in norm_group
            ):
                self.mmr_min = float(norm_group["mmr_min"][()])
                self.mmr_max = float(norm_group["mmr_max"][()])
                self.mmr_epsilon = float(norm_group["mmr_epsilon"][()])
            else:
                return False

            return True

        print("Computing new normalization stats")
        # Compute new normalization stats
        # Get training indices
        train_indices = self.indices if self.mode == "train" else None
        if train_indices is None:
            time_dim = h5f["metadata/time"].shape[0]
            shuffled_indices = (
                np.random.permutation(time_dim) if self.shuffle else np.arange(time_dim)
            )
            train_size = int(time_dim * self.split[0])
            train_indices = np.sort(shuffled_indices[:train_size])

        # Compute input stats
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

        # Compute emission stats if needed
        if self.include_emissions:
            self.emission_mean = {}
            self.emission_std = {}
            n_particles = len(self.settling_velocities)

            for size_idx in range(n_particles):
                emission_data = h5f[
                    f"outputs/emissions/Plast0{size_idx+1}_SRF_EMIS_avrg"
                ][train_indices]
                self.emission_mean[size_idx] = float(np.mean(emission_data))
                self.emission_std[size_idx] = float(np.std(emission_data))
                if self.emission_std[size_idx] == 0:
                    self.emission_std[size_idx] = 1.0

        # Compute output stats
        self.output_mean = {"mmr": {}, "dry_dep": {}, "wet_dep": {}}
        self.output_std = {"mmr": {}, "dry_dep": {}, "wet_dep": {}}

        # Compute MMR log-scale stats
        n_particles = len(self.settling_velocities)
        self.mmr_min = float("inf")
        self.mmr_max = float("-inf")
        self.mmr_epsilon = 1e-20

        for size_idx in range(n_particles):
            # MMR stats
            mmr_data = h5f[f"outputs/mass_mixing_ratio/Plast0{size_idx+1}_MMR_avrg"][
                train_indices
            ]
            self.output_mean["mmr"][size_idx] = float(np.mean(mmr_data))
            self.output_std["mmr"][size_idx] = float(np.std(mmr_data))
            if self.output_std["mmr"][size_idx] == 0:
                self.output_std["mmr"][size_idx] = 1.0

            # Log-scale MMR stats
            mmr_data = mmr_data + self.mmr_epsilon
            log_mmr_data = np.log10(mmr_data)
            self.mmr_min = min(self.mmr_min, float(np.min(log_mmr_data)))
            self.mmr_max = max(self.mmr_max, float(np.max(log_mmr_data)))

            if not self.only_mmr:
                # Deposition stats
                for dep_type, path in [
                    ("dry_dep", "deposition/Plast0{}_DRY_DEP_FLX_avrg"),
                    ("wet_dep", "deposition/Plast0{}_WETDEP_FLUX_avrg"),
                ]:
                    dep_data = h5f[f"outputs/{path.format(size_idx+1)}"][train_indices]
                    self.output_mean[dep_type][size_idx] = float(np.mean(dep_data))
                    self.output_std[dep_type][size_idx] = float(np.std(dep_data))
                    if self.output_std[dep_type][size_idx] == 0:
                        self.output_std[dep_type][size_idx] = 1.0

        # Add small epsilon to avoid division by zero
        if self.mmr_max == self.mmr_min:
            self.mmr_max += 1e-10

        # Save normalization stats
        h5f.close()
        with h5py.File(self.h5_file, "a") as h5f:
            if "normalization" in h5f:
                del h5f["normalization"]
            norm_group = h5f.create_group("normalization")

            # Save input stats
            for var_name in met_vars:
                norm_group.create_dataset(
                    f"{var_name}_mean", data=self.input_mean[var_name]
                )
                norm_group.create_dataset(
                    f"{var_name}_std", data=self.input_std[var_name]
                )

            # Save emission stats if needed
            if self.include_emissions:
                for size_idx in range(n_particles):
                    norm_group.create_dataset(
                        f"emission_{size_idx}_mean", data=self.emission_mean[size_idx]
                    )
                    norm_group.create_dataset(
                        f"emission_{size_idx}_std", data=self.emission_std[size_idx]
                    )

            # Save output stats
            for size_idx in range(n_particles):
                # MMR stats
                norm_group.create_dataset(
                    f"mmr_{size_idx}_mean", data=self.output_mean["mmr"][size_idx]
                )
                norm_group.create_dataset(
                    f"mmr_{size_idx}_std", data=self.output_std["mmr"][size_idx]
                )

                if not self.only_mmr:
                    # Deposition stats
                    for dep_type in ["dry_dep", "wet_dep"]:
                        norm_group.create_dataset(
                            f"{dep_type}_{size_idx}_mean",
                            data=self.output_mean[dep_type][size_idx],
                        )
                        norm_group.create_dataset(
                            f"{dep_type}_{size_idx}_std",
                            data=self.output_std[dep_type][size_idx],
                        )

            # Save MMR log-scale stats
            norm_group.create_dataset("mmr_min", data=self.mmr_min)
            norm_group.create_dataset("mmr_max", data=self.mmr_max)
            norm_group.create_dataset("mmr_epsilon", data=self.mmr_epsilon)

        return True

    def _open_file(self):
        """Lazily open the file handle when needed"""
        if self._file_handle is None:
            self._file_handle = h5py.File(self.h5_file, "r")
        return self._file_handle

    def __del__(self):
        """Clean up file handle when object is destroyed"""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        time_idx = self.indices[idx]
        h5f = self._open_file()

        # Input fields processing is fine
        ps = h5f["inputs/PS"][time_idx]
        u = h5f["inputs/U"][time_idx]
        v = h5f["inputs/V"][time_idx]
        t = h5f["inputs/T"][time_idx]
        q = h5f["inputs/Q"][time_idx]

        if self.include_troplev:
            troplev = h5f["inputs/TROPLEV"][time_idx]

        # Convert to torch tensors immediately
        ps = torch.from_numpy(ps).to(torch.float32)
        u = torch.from_numpy(u).to(torch.float32)
        v = torch.from_numpy(v).to(torch.float32)
        t = torch.from_numpy(t).to(torch.float32)
        q = torch.from_numpy(q).to(torch.float32)

        if self.include_troplev:
            troplev = torch.from_numpy(troplev).to(torch.float32)

        # Normalize in tensor space
        if self.normalize:
            ps = (ps - self.input_mean["PS"]) / self.input_std["PS"]
            u = (u - self.input_mean["U"]) / self.input_std["U"]
            v = (v - self.input_mean["V"]) / self.input_std["V"]
            t = (t - self.input_mean["T"]) / self.input_std["T"]
            q = (q - self.input_mean["Q"]) / self.input_std["Q"]

            if self.include_troplev:
                troplev = (troplev - self.input_mean["TROPLEV"]) / self.input_std[
                    "TROPLEV"
                ]

        # Repeat to match shape
        shape = u.shape
        ps_3d = ps.unsqueeze(0).repeat(shape[0], 1, 1)

        # Stack inputs
        input_tensors = [ps_3d, u, v, t, q]
        if self.include_troplev:
            troplev_3d = troplev.unsqueeze(0).repeat(shape[0], 1, 1)
            input_tensors.append(troplev_3d)

        # Stack all inputs
        inputs = torch.stack(input_tensors, dim=0)

        # Add emissions to inputs if needed
        if self.include_emissions:
            # Concatenate along the first dimension (channels)
            inputs = torch.cat([inputs, self.emissions], dim=0)

        # Process MMR data
        n_particles = len(self.settling_velocities)
        mmr_list = []

        for size_idx in range(n_particles):
            # Get raw MMR data as numpy array first to ensure consistency with original code
            mmr_raw = h5f[f"outputs/mass_mixing_ratio/Plast0{size_idx+1}_MMR_avrg"][
                time_idx
            ]

            if self.normalize:
                # Add epsilon in numpy domain, then convert to log
                mmr_data = np.log10(mmr_raw + self.mmr_epsilon)
                # Apply min-max normalization
                mmr_data = (mmr_data - self.mmr_min) / (self.mmr_max - self.mmr_min)
                # Convert to tensor after all numpy processing
                mmr_data = torch.from_numpy(mmr_data).to(torch.float32)
            else:
                mmr_data = torch.from_numpy(mmr_raw).to(torch.float32)

            mmr_list.append(mmr_data)

        mmr = torch.stack(mmr_list)

        # Process deposition data regardless of only_mmr flag
        dry_dep_list = []
        wet_dep_list = []

        for size_idx in range(n_particles):
            # Get raw data as numpy arrays
            dry_raw = h5f[f"outputs/deposition/Plast0{size_idx+1}_DRY_DEP_FLX_avrg"][
                time_idx
            ]
            wet_raw = h5f[f"outputs/deposition/Plast0{size_idx+1}_WETDEP_FLUX_avrg"][
                time_idx
            ]

            if self.normalize:
                # Normalize in numpy domain to ensure consistency
                if not self.only_mmr:  # Use existing normalization stats if available
                    dry_data = (
                        dry_raw - self.output_mean["dry_dep"][size_idx]
                    ) / self.output_std["dry_dep"][size_idx]
                    wet_data = (
                        wet_raw - self.output_mean["wet_dep"][size_idx]
                    ) / self.output_std["wet_dep"][size_idx]
                else:  # Just use min-max normalization if we don't have normalization stats
                    dry_data = (dry_raw - np.min(dry_raw)) / (
                        np.max(dry_raw) - np.min(dry_raw) + 1e-10
                    )
                    wet_data = (wet_raw - np.min(wet_raw)) / (
                        np.max(wet_raw) - np.min(wet_raw) + 1e-10
                    )

                # Convert to tensors after normalization
                dry_data = torch.from_numpy(dry_data).to(torch.float32)
                wet_data = torch.from_numpy(wet_data).to(torch.float32)
            else:
                dry_data = torch.from_numpy(dry_raw).to(torch.float32)
                wet_data = torch.from_numpy(wet_raw).to(torch.float32)

            # Add dimension to match expected shape for concatenation
            dry_data = dry_data.unsqueeze(0)  # shape: [1, 384, 576]
            wet_data = wet_data.unsqueeze(0)  # shape: [1, 384, 576]

            # Store for each particle size
            dry_dep_list.append(dry_data)
            wet_dep_list.append(wet_data)

        # Stack all the deposition data
        dry_dep = torch.stack(dry_dep_list)  # shape: [6, 1, 384, 576]
        wet_dep = torch.stack(wet_dep_list)  # shape: [6, 1, 384, 576]

        # Combine the deposition data
        dep = torch.cat([dry_dep, wet_dep], dim=1)  # shape: [6, 2, 384, 576]

        if self.reverse_lev:
            mmr = mmr.flip(1)
            inputs = inputs.flip(1)

        # Combine MMR and deposition data to create output with shape [6, 50, 384, 576]
        combined_output = torch.cat([mmr, dep], dim=1)  # shape: [6, 50, 384, 576]

        if not self.include_metadata:
            return inputs, combined_output
        else:
            metadata = {
                "P0": self.P0,
                "gw": self.gw,
                "hyai": self.hyai,
                "hybi": self.hybi,
                "settling_velocities": self.settling_velocities,
                "emission_mass": self.emission_mass,
            }
            return inputs, combined_output, metadata


class MPIDataModule(pl.LightningDataModule):

    def __init__(
        self,
        batch_size: int = 1,
        num_workers: int = 4,
        split: Tuple[float, float, float] = (0.5, 0.25, 0.25),
        shuffle: bool = True,
        normalize: bool = True,
        include_coordinates: bool = True,
        include_emissions: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split
        self.normalize = normalize
        self.include_coordinates = include_coordinates
        self.shuffle = shuffle
        self.include_emissions = include_emissions

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
                shuffle=self.shuffle,
                force_normalize=False,
                include_emissions=self.include_emissions,
            )

            self.val_dataset = MPIDataset(
                h5_file=self.h5_file,
                mode="val",
                split=self.split,
                normalize=self.normalize,
                include_coordinates=self.include_coordinates,
                shuffle=self.shuffle,
                force_normalize=False,
                include_emissions=self.include_emissions,
            )

        if stage == "test" or stage is None:
            self.test_dataset = MPIDataset(
                h5_file=self.h5_file,
                mode="test",
                split=self.split,
                normalize=self.normalize,
                include_coordinates=self.include_coordinates,
                shuffle=self.shuffle,
                force_normalize=False,
                include_emissions=self.include_emissions,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
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


def get_normalization_stats(h5_file: str) -> Dict:
    """Get normalization statistics from the HDF5 file.

    Args:
        h5_file: Path to the HDF5 file

    Returns:
        Dict containing all normalization statistics:
        {
            'input_mean': {'PS': float, 'U': float, ...},
            'input_std': {'PS': float, 'U': float, ...},
            'emission_mean': {0: float, 1: float, ...},
            'emission_std': {0: float, 1: float, ...},
            'output_mean': {
                'mmr': {0: float, 1: float, ...},
                'dry_dep': {0: float, 1: float, ...},
                'wet_dep': {0: float, 1: float, ...}
            },
            'output_std': {
                'mmr': {0: float, 1: float, ...},
                'dry_dep': {0: float, 1: float, ...},
                'wet_dep': {0: float, 1: float, ...}
            },
            'mmr_log_scale': {
                'min': float,
                'max': float,
                'epsilon': float
            }
        }

    Raises:
        ValueError: If normalization statistics are not found in the file
    """
    with h5py.File(h5_file, "r") as h5f:
        if "normalization" not in h5f:
            raise ValueError("Normalization statistics not found in HDF5 file")

        norm_group = h5f["normalization"]
        stats = {
            "input_mean": {},
            "input_std": {},
            "emission_mean": {},
            "emission_std": {},
            "output_mean": {"mmr": {}, "dry_dep": {}, "wet_dep": {}},
            "output_std": {"mmr": {}, "dry_dep": {}, "wet_dep": {}},
            "mmr_log_scale": {},
        }

        # Get input stats
        met_vars = ["PS", "U", "V", "T", "Q"]
        if "TROPLEV_mean" in norm_group:
            met_vars.append("TROPLEV")

        for var in met_vars:
            stats["input_mean"][var] = float(norm_group[f"{var}_mean"][()])
            stats["input_std"][var] = float(norm_group[f"{var}_std"][()])

        # Get emission stats if they exist
        n_particles = 0
        while f"emission_{n_particles}_mean" in norm_group:
            stats["emission_mean"][n_particles] = float(
                norm_group[f"emission_{n_particles}_mean"][()]
            )
            stats["emission_std"][n_particles] = float(
                norm_group[f"emission_{n_particles}_std"][()]
            )
            n_particles += 1

        # If no emission stats found, determine n_particles from MMR stats
        if n_particles == 0:
            while f"mmr_{n_particles}_mean" in norm_group:
                n_particles += 1

        # Get output stats
        for size_idx in range(n_particles):
            # MMR stats
            stats["output_mean"]["mmr"][size_idx] = float(
                norm_group[f"mmr_{size_idx}_mean"][()]
            )
            stats["output_std"]["mmr"][size_idx] = float(
                norm_group[f"mmr_{size_idx}_std"][()]
            )

            # Deposition stats if they exist
            for dep_type in ["dry_dep", "wet_dep"]:
                if f"{dep_type}_{size_idx}_mean" in norm_group:
                    stats["output_mean"][dep_type][size_idx] = float(
                        norm_group[f"{dep_type}_{size_idx}_mean"][()]
                    )
                    stats["output_std"][dep_type][size_idx] = float(
                        norm_group[f"{dep_type}_{size_idx}_std"][()]
                    )

        # Get MMR log-scale stats
        stats["mmr_log_scale"] = {
            "min": float(norm_group["mmr_min"][()]),
            "max": float(norm_group["mmr_max"][()]),
            "epsilon": float(norm_group["mmr_epsilon"][()]),
        }

        return stats


if __name__ == "__main__":
    from src.utils.memory import print_memory_allocated, print_tensor_memory

    print_memory_allocated()

    datamodule = MPIDataModule(shuffle=False)
    datamodule.setup()

    print_memory_allocated()

    train_dataloader = datamodule.train_dataloader()

    print(len(train_dataloader))

    for batch in train_dataloader:
        X, y, metadata = batch
        print(X.shape)  # [1, 11, 48, 384, 576]
        print_tensor_memory(X)
        print(y.shape)  # [1, 6, 50, 384, 576]
        print_tensor_memory(y)
        print_memory_allocated()

        print(f"PS range: {X[0, 0].min()} - {X[0, 0].max()}")
        print(f"U range: {X[0, 1].min()} - {X[0, 1].max()}")
        print(f"V range: {X[0, 2].min()} - {X[0, 2].max()}")
        print(f"T range: {X[0, 3].min()} - {X[0, 3].max()}")
        print(f"Q range: {X[0, 4].min()} - {X[0, 4].max()}")
        print(f"Source range: {X[0, 5:11].min()} - {X[0, 5:11].max()}")

        print(f"MMR range: {y[0, :, :48].min()} - {y[0, :, :48].max()}")
        print(f"Dry deposition range: {y[0, :, 48].min()} - {y[0, :, 48].max()}")
        print(f"Wet deposition range: {y[0, :, 49].min()} - {y[0, :, 49].max()}")

        for key, value in metadata.items():
            print(f"{key}: {value.shape}")

        # P0: torch.Size([1])
        # gw: torch.Size([1, 384])
        # hyai: torch.Size([1, 49])
        # hybi: torch.Size([1, 49])
        # settling_velocities: torch.Size([1, 6])
        # emission_mass: torch.Size([1])

        break
