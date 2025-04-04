import os
import h5py
import numpy as np
from netCDF4 import Dataset
from datetime import datetime
import glob
from src.utils import PATH
from tqdm import tqdm


def process_netcdf_files():
    """
    Aggregate 24 monthly NetCDF files from 2013-01 to 2014-12 into a single HDF5 file
    organized for PINN training with meteorological inputs and microplastic outputs.
    """

    input_dir = PATH.RAW_DATA / "mpidata"
    output_file = PATH.PROCESSED_DATA / "mpidata.h5"

    # The ending is yyyy-mm
    file_pattern = "MZ4_plastic_6sp_wd_oldsettl.mz4.h0.*.nc"

    input_files = sorted(glob.glob(os.path.join(input_dir, file_pattern)))

    if not input_files:
        raise FileNotFoundError(
            f"No files matching {file_pattern} found in {input_dir}"
        )

    print(f"Found {len(input_files)} files to process")

    os.makedirs(PATH.PROCESSED_DATA, exist_ok=True)

    # Pressure, wind East, wind North, temperature, humidity, tropopause
    met_vars = ["PS", "U", "V", "T", "Q", "TROPLEV"]

    # Microplastic mass mixing ratio
    mp_mmr_vars = [f"Plast0{i}_MMR_avrg" for i in range(1, 7)]

    # Surface emission (source) terms
    mp_src_vars = [f"Plast0{i}_SRF_EMIS_avrg" for i in range(1, 7)]

    # Microplastic deposition fluxes
    mp_dep_vars = [
        f"Plast0{i}_DRY_DEP_FLX_avrg" for i in range(1, 7)
    ] + [  # Dry deposition
        f"Plast0{i}_WETDEP_FLUX_avrg" for i in range(1, 7)
    ]  # Wet deposition

    with h5py.File(output_file, "w") as hf:

        # Metadata
        hf.attrs["description"] = (
            "Microplastic atmospheric transport data for PINN training"
        )
        hf.attrs["creation_date"] = datetime.now().isoformat()
        hf.attrs["source"] = "MZ4_plastic_6sp_wd_oldsettl NetCDF files"
        hf.attrs["time_range"] = "2013-01 to 2014-12"

        # Groups
        metadata_grp = hf.create_group("metadata")

        input_grp = hf.create_group("inputs")  # Meteorological data
        output_grp = hf.create_group("outputs")  # Microplastic data

        # First file for reference
        print("Setting up coordinate data...")
        with Dataset(input_files[0], "r") as nc:

            # Datasets
            metadata_grp.create_dataset("lon", data=nc.variables["lon"][:])
            metadata_grp.create_dataset("lat", data=nc.variables["lat"][:])
            metadata_grp.create_dataset("lev", data=nc.variables["lev"][:])

            # Coordinates
            for coord in ["lon", "lat", "lev"]:
                for attr_name in nc.variables[coord].ncattrs():
                    # Skip _FillValue
                    if attr_name != "_FillValue":
                        metadata_grp[coord].attrs[attr_name] = getattr(
                            nc.variables[coord], attr_name
                        )

            # Vertical coordinate transformations
            metadata_grp.create_dataset(
                "hyam", data=nc.variables["hyam"][:]
            )  # Hybrid A
            metadata_grp.create_dataset(
                "hybm", data=nc.variables["hybm"][:]
            )  # Hybrid B
            metadata_grp.create_dataset(
                "P0", data=nc.variables["P0"][:]
            )  # Reference pressure

            # Dimensions
            time_dim = len(input_files)
            lon_dim = len(nc.dimensions["lon"])
            lat_dim = len(nc.dimensions["lat"])
            lev_dim = len(nc.dimensions["lev"])

            # Timestamps
            timestamps = metadata_grp.create_dataset(
                "time", shape=(time_dim,), dtype="S10"
            )

            # Settling velocity (m/s)
            metadata_grp.create_dataset(
                "settling_velocity",
                data=np.array([0.00097, 0.0087, 0.097, 0.39, 2.7, 4.98]),
                dtype="float32",
            )

        # Pre-allocation

        # Pressure and tropopause is 3D
        # Wind (East, North), temperature, humidity is 4D (with altitude)
        print("Creating datasets...")
        for var in met_vars:

            if var == "PS" or var == "TROPLEV":
                input_grp.create_dataset(
                    var, shape=(time_dim, lat_dim, lon_dim), dtype="float32"
                )

            else:
                input_grp.create_dataset(
                    var, shape=(time_dim, lev_dim, lat_dim, lon_dim), dtype="float32"
                )

        # MP MMR is 4D
        mp_mmr_grp = output_grp.create_group("mass_mixing_ratio")
        for var in mp_mmr_vars:
            mp_mmr_grp.create_dataset(
                var, shape=(time_dim, lev_dim, lat_dim, lon_dim), dtype="float32"
            )

        # MP surface emissions are 3D
        mp_src_grp = output_grp.create_group("emissions")
        for var in mp_src_vars:
            mp_src_grp.create_dataset(
                var, shape=(time_dim, lat_dim, lon_dim), dtype="float32"
            )

        # MP deposition fluxes are 3D
        mp_dep_grp = output_grp.create_group("deposition")
        for var in mp_dep_vars:
            mp_dep_grp.create_dataset(
                var, shape=(time_dim, lat_dim, lon_dim), dtype="float32"
            )

        # Process each file
        for i, file_path in enumerate(tqdm(input_files, desc="Processing files")):
            # Extract timestamp from filename (yyyy-mm)
            timestamp = os.path.basename(file_path).split(".")[-2]
            timestamps[i] = timestamp

            with Dataset(file_path, "r") as nc:
                # Refer to pre-allocating for docs

                for var in met_vars:

                    if var == "PS" or var == "TROPLEV":
                        input_grp[var][i] = nc.variables[var][0]
                    else:
                        input_grp[var][i] = nc.variables[var][0]

                    if i == 0:
                        for attr_name in nc.variables[var].ncattrs():
                            if attr_name != "_FillValue":
                                input_grp[var].attrs[attr_name] = getattr(
                                    nc.variables[var], attr_name
                                )

                for var in mp_mmr_vars:
                    mp_mmr_grp[var][i] = nc.variables[var][0]

                    # Copy attributes on first iteration
                    if i == 0:
                        for attr_name in nc.variables[var].ncattrs():
                            if attr_name != "_FillValue":
                                mp_mmr_grp[var].attrs[attr_name] = getattr(
                                    nc.variables[var], attr_name
                                )

                for var in mp_src_vars:
                    mp_src_grp[var][i] = nc.variables[var][0]

                    # Copy attributes on first iteration
                    if i == 0:
                        for attr_name in nc.variables[var].ncattrs():
                            if attr_name != "_FillValue":
                                mp_src_grp[var].attrs[attr_name] = getattr(
                                    nc.variables[var], attr_name
                                )

                for var in mp_dep_vars:
                    mp_dep_grp[var][i] = nc.variables[var][0]

                    # Copy attributes on first iteration
                    if i == 0:
                        for attr_name in nc.variables[var].ncattrs():
                            if attr_name != "_FillValue":
                                mp_dep_grp[var].attrs[attr_name] = getattr(
                                    nc.variables[var], attr_name
                                )

    print(f"Processing complete. Data saved to {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024*1024):.2f} GB")


if __name__ == "__main__":
    process_netcdf_files()
