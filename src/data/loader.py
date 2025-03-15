import h5py
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Generator
from datetime import datetime


@dataclass
class MPIDataset:
    time: np.ndarray
    date: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    lev: np.ndarray
    ilev: np.ndarray
    vertical_coords: Dict[str, np.ndarray]
    dynamics: Dict[str, np.ndarray]
    small_particles: Dict[str, np.ndarray]
    large_particles: Dict[str, np.ndarray]
    emissions: Dict[str, np.ndarray]
    dry_deposition: Dict[str, np.ndarray]
    wet_deposition: Dict[str, np.ndarray]
    metadata: Dict[str, str]
    _file_handle: h5py.File

    def __del__(self):
        """Cleanup file handle when object is destroyed"""
        if hasattr(self, "_file_handle") and self._file_handle:
            self._file_handle.close()


class MPIDataLoader:
    def __init__(self, file_path: Union[str, Path], cache_size: int = 32):
        self.file_path = Path(file_path)
        self.cache_size = cache_size
        if not self.file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.file_path}")
        # Keep file handle open for faster repeated access
        self._file = h5py.File(
            self.file_path, "r", swmr=True
        )  # SWMR mode for better performance

    def __del__(self):
        """Cleanup file handle when object is destroyed"""
        if hasattr(self, "_file"):
            self._file.close()

    def load_full_dataset(self, lazy: bool = True) -> MPIDataset:
        if lazy:
            return MPIDataset(
                time=self._file[
                    "coordinates/time"
                ],  # Return h5py dataset instead of loading
                date=self._file["coordinates/date"],
                lon=self._file["coordinates/lon"],
                lat=self._file["coordinates/lat"],
                lev=self._file["coordinates/lev"],
                ilev=self._file["coordinates/ilev"],
                vertical_coords=self._load_vertical_coords(self._file),
                dynamics=self._create_lazy_dict(self._file["physics/dynamics"]),
                small_particles=self._create_lazy_dict(
                    self._file["physics/microplastics/small_particles"]
                ),
                large_particles=self._create_lazy_dict(
                    self._file["physics/microplastics/large_particles"]
                ),
                emissions=self._create_lazy_dict(
                    self._file["boundary_conditions/surface_emissions"]
                ),
                dry_deposition=self._create_lazy_dict(
                    self._file["boundary_conditions/deposition_rates/dry_deposition"]
                ),
                wet_deposition=self._create_lazy_dict(
                    self._file["boundary_conditions/deposition_rates/wet_deposition"]
                ),
                metadata=self._load_metadata(self._file),
                _file_handle=self._file,
            )
        else:
            with h5py.File(self.file_path, "r") as f:
                return MPIDataset(
                    time=f["coordinates/time"][:],
                    date=f["coordinates/date"][:],
                    lon=f["coordinates/lon"][:],
                    lat=f["coordinates/lat"][:],
                    lev=f["coordinates/lev"][:],
                    ilev=f["coordinates/ilev"][:],
                    vertical_coords=self._load_vertical_coords(f),
                    dynamics=self._load_group_data(f["physics/dynamics"]),
                    small_particles=self._load_group_data(
                        f["physics/microplastics/small_particles"]
                    ),
                    large_particles=self._load_group_data(
                        f["physics/microplastics/large_particles"]
                    ),
                    emissions=self._load_group_data(
                        f["boundary_conditions/surface_emissions"]
                    ),
                    dry_deposition=self._load_group_data(
                        f["boundary_conditions/deposition_rates/dry_deposition"]
                    ),
                    wet_deposition=self._load_group_data(
                        f["boundary_conditions/deposition_rates/wet_deposition"]
                    ),
                    metadata=self._load_metadata(f),
                )

    def load_time_slice(self, time_idx: int) -> MPIDataset:
        with h5py.File(self.file_path, "r") as f:
            return MPIDataset(
                time=f["coordinates/time"][time_idx : time_idx + 1],
                date=f["coordinates/date"][time_idx : time_idx + 1],
                lon=f["coordinates/lon"][:],
                lat=f["coordinates/lat"][:],
                lev=f["coordinates/lev"][:],
                ilev=f["coordinates/ilev"][:],
                vertical_coords=self._load_vertical_coords(f),
                dynamics=self._load_group_data(f["physics/dynamics"], time_idx),
                small_particles=self._load_group_data(
                    f["physics/microplastics/small_particles"], time_idx
                ),
                large_particles=self._load_group_data(
                    f["physics/microplastics/large_particles"], time_idx
                ),
                emissions=self._load_group_data(
                    f["boundary_conditions/surface_emissions"], time_idx
                ),
                dry_deposition=self._load_group_data(
                    f["boundary_conditions/deposition_rates/dry_deposition"], time_idx
                ),
                wet_deposition=self._load_group_data(
                    f["boundary_conditions/deposition_rates/wet_deposition"], time_idx
                ),
                metadata=self._load_metadata(f),
            )

    def _load_vertical_coords(self, f: h5py.File) -> Dict[str, np.ndarray]:
        vert_group = f["coordinates/vertical_coords"]
        return {
            "hyam": vert_group["hyam"][:],
            "hybm": vert_group["hybm"][:],
            "hyai": vert_group["hyai"][:],
            "hybi": vert_group["hybi"][:],
            "P0": vert_group.attrs["P0"],
        }

    def _load_group_data(
        self, group: h5py.Group, time_idx: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        data = {}
        for key in group.keys():
            if time_idx is not None:
                data[key] = group[key][time_idx : time_idx + 1]
            else:
                data[key] = group[key][:]
        return data

    def _load_metadata(self, f: h5py.File) -> Dict[str, str]:
        return dict(f["metadata"].attrs)

    def get_time_range(self) -> Tuple[datetime, datetime]:
        with h5py.File(self.file_path, "r") as f:
            dates = f["coordinates/date"][:]
            start_date = datetime.strptime(str(dates[0]), "%Y%m%d")
            end_date = datetime.strptime(str(dates[-1]), "%Y%m%d")
            return start_date, end_date

    def get_variable_names(self) -> Dict[str, List[str]]:
        with h5py.File(self.file_path, "r") as f:
            return {
                "dynamics": list(f["physics/dynamics"].keys()),
                "small_particles": list(
                    f["physics/microplastics/small_particles"].keys()
                ),
                "large_particles": list(
                    f["physics/microplastics/large_particles"].keys()
                ),
                "emissions": list(f["boundary_conditions/surface_emissions"].keys()),
                "dry_deposition": list(
                    f["boundary_conditions/deposition_rates/dry_deposition"].keys()
                ),
                "wet_deposition": list(
                    f["boundary_conditions/deposition_rates/wet_deposition"].keys()
                ),
            }

    def _create_lazy_dict(self, group: h5py.Group) -> Dict[str, h5py.Dataset]:
        """Create a dictionary of dataset references without loading data"""
        return {key: group[key] for key in group.keys()}

    def load_chunked(self, chunk_size: int = 100) -> Generator[MPIDataset, None, None]:
        """Load the dataset in chunks"""
        total_times = self._file["coordinates/time"].shape[0]
        for start_idx in range(0, total_times, chunk_size):
            end_idx = min(start_idx + chunk_size, total_times)
            yield self.load_time_slice(slice(start_idx, end_idx))
