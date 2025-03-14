import os
import h5py
import numpy as np
from netCDF4 import Dataset
from pathlib import Path
import time
from datetime import datetime
from tqdm import tqdm

# Import path from utils
from src.utils import PATH


def create_pinn_hdf5_dataset():
    """
    Process 24 monthly MOZART4 NetCDF files into a single HDF5 file
    optimized for Physics-Informed Neural Network analysis.
    """
    start_time = time.time()
    print("Starting microplastic data processing...")

    # Define input and output paths
    output_file = PATH.PROCESSED_DATA / "mpidata.h5"

    # Define time range
    years = [2013, 2014]
    months = [f"{m:02d}" for m in range(1, 13)]

    # Get first file to extract dimensions and metadata
    first_file = (
        PATH.RAW_DATA
        / "mpidata"
        / f"MZ4_plastic_6sp_wd_oldsettl.mz4.h0.{years[0]}-{months[0]}.nc"
    )

    # Check if file exists
    if not first_file.exists():
        raise FileNotFoundError(f"File not found: {first_file}")

    print(f"Reading dimensions from {first_file}")

    # Extract dimensions and metadata from first file
    with Dataset(first_file, "r") as nc:
        # Get dimensions
        lon = nc.variables["lon"][:]
        lat = nc.variables["lat"][:]
        lev = nc.variables["lev"][:]
        ilev = nc.variables["ilev"][:]

        # Get hybrid coefficients for vertical coordinate
        hyam = nc.variables["hyam"][:]
        hybm = nc.variables["hybm"][:]
        hyai = nc.variables["hyai"][:]
        hybi = nc.variables["hybi"][:]
        P0 = nc.variables["P0"][...]

        # Get variable names
        all_vars = list(nc.variables.keys())

        # Identify microplastic variables
        plast_mmr_vars = [
            var
            for var in all_vars
            if var.startswith("Plast") and var.endswith("MMR_avrg")
        ]
        plast_emis_vars = [
            var
            for var in all_vars
            if var.startswith("Plast") and var.endswith("SRF_EMIS_avrg")
        ]
        plast_dry_dep_vars = [
            var
            for var in all_vars
            if var.startswith("Plast") and var.endswith("DRY_DEP_FLX_avrg")
        ]
        plast_wet_dep_vars = [
            var
            for var in all_vars
            if var.startswith("Plast") and var.endswith("WETDEP_FLUX_avrg")
        ]

        # Identify meteorological variables
        met_vars = ["PS", "U", "V", "T", "Q", "TROPLEV"]

        # Get global attributes
        global_attrs = {attr: getattr(nc, attr) for attr in nc.ncattrs()}

    # Define small and large particle variables (based on size bins)
    small_particles = plast_mmr_vars[:4]  # First 4 size bins (0.5-10 μm)
    large_particles = plast_mmr_vars[4:]  # Last 2 size bins (35-70 μm)

    # Create HDF5 file with chunking and compression
    print(f"Creating HDF5 file: {output_file}")
    with h5py.File(output_file, "w") as h5f:
        # Store metadata
        meta_grp = h5f.create_group("metadata")
        meta_grp.attrs["creation_date"] = datetime.now().isoformat()
        meta_grp.attrs["source_files"] = (
            f"MOZART4 microplastic simulations {years[0]}-{years[-1]}"
        )
        meta_grp.attrs["description"] = "Processed microplastic data for PINN analysis"

        # Add global attributes from NetCDF
        for key, value in global_attrs.items():
            meta_grp.attrs[key] = value

        # Store coordinates
        coords_grp = h5f.create_group("coordinates")
        coords_grp.create_dataset(
            "lon", data=lon, compression="gzip", compression_opts=4
        )
        coords_grp.create_dataset(
            "lat", data=lat, compression="gzip", compression_opts=4
        )
        coords_grp.create_dataset(
            "lev", data=lev, compression="gzip", compression_opts=4
        )
        coords_grp.create_dataset(
            "ilev", data=ilev, compression="gzip", compression_opts=4
        )

        # Store vertical coordinate information
        vert_grp = coords_grp.create_group("vertical_coords")
        vert_grp.create_dataset("hyam", data=hyam)
        vert_grp.create_dataset("hybm", data=hybm)
        vert_grp.create_dataset("hyai", data=hyai)
        vert_grp.create_dataset("hybi", data=hybi)
        vert_grp.attrs["P0"] = P0

        # Create time dimension
        time_dim = len(years) * len(months)
        time_data = np.zeros(time_dim, dtype=np.float64)
        date_data = np.zeros(time_dim, dtype=np.int32)

        # Create physics group
        physics_grp = h5f.create_group("physics")

        # Create dynamics subgroup for meteorological variables
        dyn_grp = physics_grp.create_group("dynamics")

        # Create microplastics subgroup
        mp_grp = physics_grp.create_group("microplastics")
        small_grp = mp_grp.create_group("small_particles")
        large_grp = mp_grp.create_group("large_particles")

        # Create boundary conditions group
        bc_grp = h5f.create_group("boundary_conditions")
        emissions_grp = bc_grp.create_group("surface_emissions")
        deposition_grp = bc_grp.create_group("deposition_rates")
        dry_dep_grp = deposition_grp.create_group("dry_deposition")
        wet_dep_grp = deposition_grp.create_group("wet_deposition")

        # Create derivatives group
        deriv_grp = h5f.create_group("derivatives")

        # Initialize datasets for meteorological variables
        for var in met_vars:
            if var == "PS" or var == "TROPLEV":
                # 2D variables (time, lat, lon)
                dyn_grp.create_dataset(
                    var,
                    shape=(time_dim, len(lat), len(lon)),
                    chunks=(1, len(lat), len(lon)),
                    compression="gzip",
                    compression_opts=4,
                    dtype=np.float32,
                )
            else:
                # 3D variables (time, lev, lat, lon)
                dyn_grp.create_dataset(
                    var,
                    shape=(time_dim, len(lev), len(lat), len(lon)),
                    chunks=(1, len(lev), len(lat), len(lon)),
                    compression="gzip",
                    compression_opts=4,
                    dtype=np.float32,
                )

        # Initialize datasets for microplastic variables
        for var in small_particles:
            small_grp.create_dataset(
                var,
                shape=(time_dim, len(lev), len(lat), len(lon)),
                chunks=(1, len(lev), len(lat), len(lon)),
                compression="gzip",
                compression_opts=4,
                dtype=np.float32,
            )

        for var in large_particles:
            large_grp.create_dataset(
                var,
                shape=(time_dim, len(lev), len(lat), len(lon)),
                chunks=(1, len(lev), len(lat), len(lon)),
                compression="gzip",
                compression_opts=4,
                dtype=np.float32,
            )

        # Initialize datasets for emissions
        for var in plast_emis_vars:
            emissions_grp.create_dataset(
                var,
                shape=(time_dim, len(lat), len(lon)),
                chunks=(1, len(lat), len(lon)),
                compression="gzip",
                compression_opts=4,
                dtype=np.float32,
            )

        # Initialize datasets for deposition
        for var in plast_dry_dep_vars:
            dry_dep_grp.create_dataset(
                var,
                shape=(time_dim, len(lat), len(lon)),
                chunks=(1, len(lat), len(lon)),
                compression="gzip",
                compression_opts=4,
                dtype=np.float32,
            )

        for var in plast_wet_dep_vars:
            wet_dep_grp.create_dataset(
                var,
                shape=(time_dim, len(lat), len(lon)),
                chunks=(1, len(lat), len(lon)),
                compression="gzip",
                compression_opts=4,
                dtype=np.float32,
            )

        # Process each file
        time_idx = 0
        for year in years:
            for month in months:
                file_path = (
                    PATH.RAW_DATA
                    / "mpidata"
                    / f"MZ4_plastic_6sp_wd_oldsettl.mz4.h0.{year}-{month}.nc"
                )

                if not file_path.exists():
                    print(f"Warning: File not found: {file_path}")
                    continue

                print(f"Processing {file_path}")

                with Dataset(file_path, "r") as nc:
                    # Store time information
                    time_data[time_idx] = nc.variables["time"][0]
                    date_data[time_idx] = nc.variables["date"][0]

                    # Store meteorological variables
                    for var in met_vars:
                        if var == "PS" or var == "TROPLEV":
                            dyn_grp[var][time_idx, :, :] = nc.variables[var][0, :, :]
                        else:
                            dyn_grp[var][time_idx, :, :, :] = nc.variables[var][
                                0, :, :, :
                            ]

                    # Store microplastic variables
                    for var in small_particles:
                        small_grp[var][time_idx, :, :, :] = nc.variables[var][
                            0, :, :, :
                        ]

                    for var in large_particles:
                        large_grp[var][time_idx, :, :, :] = nc.variables[var][
                            0, :, :, :
                        ]

                    # Store emission variables
                    for var in plast_emis_vars:
                        emissions_grp[var][time_idx, :, :] = nc.variables[var][0, :, :]

                    # Store deposition variables
                    for var in plast_dry_dep_vars:
                        dry_dep_grp[var][time_idx, :, :] = nc.variables[var][0, :, :]

                    for var in plast_wet_dep_vars:
                        wet_dep_grp[var][time_idx, :, :] = nc.variables[var][0, :, :]

                # Calculate and store derivatives (simplified example)
                # In a real implementation, you would compute proper spatial and temporal gradients

                time_idx += 1

        # Store time information
        coords_grp.create_dataset("time", data=time_data)
        coords_grp.create_dataset("date", data=date_data)

        # Calculate and store settling velocities
        # This is a placeholder - in a real implementation you would compute these based on physics
        small_grp.create_dataset(
            "settling_velocity", shape=(len(lev),), dtype=np.float32
        )
        large_grp.create_dataset(
            "settling_velocity", shape=(len(lev),), dtype=np.float32
        )

        # Set settling velocities based on particle size
        # These are placeholder values - replace with actual physics-based calculations
        small_grp["settling_velocity"][:] = np.linspace(0.001, 0.01, len(lev))  # m/s
        large_grp["settling_velocity"][:] = np.linspace(0.01, 0.1, len(lev))  # m/s

        # Add physics parameters
        physics_grp.attrs["gravity"] = 9.81  # m/s²
        physics_grp.attrs["air_viscosity"] = 1.8e-5  # kg/(m·s)
        physics_grp.attrs["particle_density"] = 1000.0  # kg/m³

        # Add size bin information
        small_grp.attrs["size_bins"] = [0.5, 1.5, 5.0, 10.0]  # μm
        small_grp.attrs["shape"] = "spherical"
        large_grp.attrs["size_bins"] = [35.0, 70.0]  # μm
        large_grp.attrs["shape"] = "fiber"

    elapsed_time = time.time() - start_time
    print(f"Processing complete. Total time: {elapsed_time:.2f} seconds")
    print(f"Output file: {output_file}")


if __name__ == "__main__":
    create_pinn_hdf5_dataset()
