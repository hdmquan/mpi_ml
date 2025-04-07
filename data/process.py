import os
import h5py
import numpy as np
from netCDF4 import Dataset
from datetime import datetime
import glob
from pathlib import Path
from src.utils import PATH
from tqdm import tqdm
from typing import List, Dict, Tuple, Any


def get_variable_groups() -> Dict[str, List[str]]:
    """Return dictionary of variable groups and their corresponding variable names."""
    return {
        "met_vars": ["PS", "U", "V", "T", "Q", "TROPLEV"],
        "mp_mmr_vars": [f"Plast0{i}_MMR_avrg" for i in range(1, 7)],
        "mp_src_vars": [f"Plast0{i}_SRF_EMIS_avrg" for i in range(1, 7)],
        "mp_dep_vars": [
            *[f"Plast0{i}_DRY_DEP_FLX_avrg" for i in range(1, 7)],
            *[f"Plast0{i}_WETDEP_FLUX_avrg" for i in range(1, 7)],
        ],
    }


def setup_metadata(hf: h5py.File, nc: Dataset) -> Tuple[h5py.Group, Dict[str, int]]:
    """Setup metadata and coordinate information in HDF5 file."""
    metadata_grp = hf.create_group("metadata")

    # Store coordinates and their attributes
    for coord in ["lon", "lat", "lev", "gw"]:
        dataset = metadata_grp.create_dataset(coord, data=nc.variables[coord][:])
        for attr_name, attr_value in nc.variables[coord].__dict__.items():
            if attr_name != "_FillValue":
                dataset.attrs[attr_name] = attr_value

    # Store vertical coordinate transformations
    for var in ["hyam", "hybm", "hyai", "hybi", "P0"]:
        metadata_grp.create_dataset(var, data=nc.variables[var][:])

    # Get dimensions
    dims = {
        "time": len(
            glob.glob(
                os.path.join(
                    PATH.RAW_DATA / "mpidata", "MZ4_plastic_6sp_wd_oldsettl.mz4.h0.*.nc"
                )
            )
        ),
        "lon": len(nc.dimensions["lon"]),
        "lat": len(nc.dimensions["lat"]),
        "lev": len(nc.dimensions["lev"]),
    }

    # Store settling velocity
    metadata_grp.create_dataset(
        "settling_velocity",
        data=np.array([0.00097, 0.0087, 0.097, 0.39, 2.7, 4.98]),
        dtype="float32",
    )

    return metadata_grp, dims


def create_variable_datasets(
    hf: h5py.File, dims: Dict[str, int], var_groups: Dict[str, List[str]]
) -> Dict[str, h5py.Group]:
    """Create datasets for all variable groups."""
    input_grp = hf.create_group("inputs")
    output_grp = hf.create_group("outputs")

    # Create meteorological datasets
    for var in var_groups["met_vars"]:
        shape = (
            (dims["time"], dims["lat"], dims["lon"])
            if var in ["PS", "TROPLEV"]
            else (dims["time"], dims["lev"], dims["lat"], dims["lon"])
        )
        input_grp.create_dataset(var, shape=shape, dtype="float32")

    # Create microplastic datasets
    group_configs = {
        "mass_mixing_ratio": ("mp_mmr_vars", True),
        "emissions": ("mp_src_vars", False),
        "deposition": ("mp_dep_vars", False),
    }

    for group_name, (var_key, include_lev) in group_configs.items():
        group = output_grp.create_group(group_name)
        base_shape = (
            dims["time"],
            dims["lev"] if include_lev else None,
            dims["lat"],
            dims["lon"],
        )
        shape = tuple(dim for dim in base_shape if dim is not None)

        for var in var_groups[var_key]:
            group.create_dataset(var, shape=shape, dtype="float32")

    return {"input": input_grp, "output": output_grp}


def copy_variable_data(
    dst_group: h5py.Group, src_var: Any, time_idx: int, copy_attrs: bool = False
) -> None:
    """Copy variable data and optionally its attributes."""
    dst_group[time_idx] = src_var[0]

    if copy_attrs:
        for attr_name, attr_value in src_var.__dict__.items():
            if attr_name != "_FillValue":
                dst_group.attrs[attr_name] = attr_value


def process_netcdf_files() -> None:
    """
    Aggregate 24 monthly NetCDF files from 2013-01 to 2014-12 into a single HDF5 file
    organized for PINN training with meteorological inputs and microplastic outputs.
    """
    input_dir = PATH.RAW_DATA / "mpidata"
    output_file = PATH.PROCESSED_DATA / "mpidata.h5"
    file_pattern = "MZ4_plastic_6sp_wd_oldsettl.mz4.h0.*.nc"

    input_files = sorted(glob.glob(os.path.join(input_dir, file_pattern)))
    if not input_files:
        raise FileNotFoundError(
            f"No files matching {file_pattern} found in {input_dir}"
        )

    print(f"Found {len(input_files)} files to process")
    os.makedirs(PATH.PROCESSED_DATA, exist_ok=True)

    var_groups = get_variable_groups()

    with h5py.File(output_file, "w") as hf:
        # Set file metadata
        hf.attrs.update(
            {
                "description": "Microplastic atmospheric transport data for PINN training",
                "creation_date": datetime.now().isoformat(),
                "source": "MZ4_plastic_6sp_wd_oldsettl NetCDF files",
                "time_range": "2013-01 to 2014-12",
            }
        )

        # Setup initial structure using first file
        with Dataset(input_files[0], "r") as nc:
            metadata_grp, dims = setup_metadata(hf, nc)
            groups = create_variable_datasets(hf, dims, var_groups)

        # Create timestamps dataset
        timestamps = metadata_grp.create_dataset(
            "time", shape=(dims["time"],), dtype="S10"
        )

        # Process each file
        for i, file_path in enumerate(tqdm(input_files, desc="Processing files")):
            timestamps[i] = os.path.basename(file_path).split(".")[
                -2
            ]  # Extract yyyy-mm

            with Dataset(file_path, "r") as nc:
                # Process meteorological variables
                for var in var_groups["met_vars"]:
                    copy_variable_data(
                        groups["input"][var], nc.variables[var], i, i == 0
                    )

                # Process microplastic variables
                for group_name, var_list in [
                    ("mass_mixing_ratio", var_groups["mp_mmr_vars"]),
                    ("emissions", var_groups["mp_src_vars"]),
                    ("deposition", var_groups["mp_dep_vars"]),
                ]:
                    for var in var_list:
                        copy_variable_data(
                            groups["output"][group_name][var],
                            nc.variables[var],
                            i,
                            i == 0,
                        )

    print(f"Processing complete. Data saved to {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024*1024):.2f} GB")


if __name__ == "__main__":
    process_netcdf_files()
