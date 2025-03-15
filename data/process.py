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


def create_dataset():
    """
    Process 24 monthly MOZART4 NetCDF files into a single HDF5 file
    """
    start_time = time.time()
    print("Starting microplastic data processing...")

    # Define input and output paths
    output_file = PATH.PROCESSED_DATA / "mpidata_v2.h5"

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
        gw = nc.variables["gw"][:]  # Gaussian weights for latitude grid

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

        # Pressure, wind, temperature, humidity, tropopause
        met_vars = ["PS", "U", "V", "T", "Q", "TROPLEV"]

        global_attrs = {attr: getattr(nc, attr) for attr in nc.ncattrs()}

    # μm
    particle_sizes = {
        "Plast01": 0.5,
        "Plast02": 1.5,
        "Plast03": 5.0,
        "Plast04": 10.0,
        "Plast05": 35.0,
        "Plast06": 70.0,
    }

    small_particles = plast_mmr_vars[:4]  # First 4 size bins (0.5-10 μm)
    large_particles = plast_mmr_vars[4:]  # Last 2 size bins (35-70 μm)

    print(f"Creating HDF5 file: {output_file}")
    with h5py.File(output_file, "w") as h5f:

        meta_grp = h5f.create_group("metadata")
        meta_grp.attrs["creation_date"] = datetime.now().isoformat()
        meta_grp.attrs["source_files"] = (
            f"MOZART4 microplastic simulations {years[0]}-{years[-1]}"
        )
        meta_grp.attrs["description"] = "Processed microplastic data"

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
        coords_grp.create_dataset("gw", data=gw, compression="gzip", compression_opts=4)

        # Add coordinate metadata
        coords_grp["lon"].attrs["units"] = "degrees_east"
        coords_grp["lat"].attrs["units"] = "degrees_north"
        coords_grp["lev"].attrs["units"] = "hybrid_sigma_pressure"
        coords_grp["gw"].attrs["description"] = "Gaussian weights for latitude grid"

        earth_radius = 6371.0  # km
        cell_areas = np.zeros((len(lat), len(lon)), dtype=np.float32)

        # Calculate approximate cell areas using Gaussian weights
        for i in range(len(lat)):
            cell_areas[i, :] = gw[i] * (2 * np.pi * earth_radius**2) / len(lon)

        coords_grp.create_dataset(
            "cell_area", data=cell_areas, compression="gzip", compression_opts=4
        )
        coords_grp["cell_area"].attrs["units"] = "km^2"
        coords_grp["cell_area"].attrs["description"] = "Approximate grid cell area"

        # Store vertical coordinate information
        vert_grp = coords_grp.create_group("vertical_coords")
        vert_grp.create_dataset("hyam", data=hyam)
        vert_grp.create_dataset("hybm", data=hybm)
        vert_grp.create_dataset("hyai", data=hyai)
        vert_grp.create_dataset("hybi", data=hybi)
        vert_grp.attrs["P0"] = P0
        vert_grp.attrs["description"] = (
            "Hybrid sigma-pressure vertical coordinate parameters"
        )

        # Create time dimension
        time_dim = len(years) * len(months)
        time_data = np.zeros(time_dim, dtype=np.float64)
        date_data = np.zeros(time_dim, dtype=np.int32)

        month_data = np.zeros(time_dim, dtype=np.int32)
        year_data = np.zeros(time_dim, dtype=np.int32)

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

        # Create ML features group
        ml_grp = h5f.create_group("ml_features")

        # Create normalized versions of key variables for ML
        norm_grp = ml_grp.create_group("normalized")

        # Create statistics group for ML preprocessing
        stats_grp = ml_grp.create_group("statistics")

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
                # Create normalized version for ML
                norm_grp.create_dataset(
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
                    compression="gzip",
                    compression_opts=4,
                    dtype=np.float32,
                )
                norm_grp.create_dataset(
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
            norm_grp.create_dataset(
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
            norm_grp.create_dataset(
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
            norm_grp.create_dataset(
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

        # Create combined total deposition dataset (useful for ML)
        deposition_grp.create_dataset(
            "total_deposition",
            shape=(time_dim, len(plast_mmr_vars), len(lat), len(lon)),
            chunks=(1, len(plast_mmr_vars), len(lat), len(lon)),
            compression="gzip",
            compression_opts=4,
            dtype=np.float32,
        )

        # Create statistics datasets for normalization
        for var_type in ["mmr", "emis", "dep"]:
            stats_grp.create_dataset(
                f"{var_type}_mean", shape=(len(plast_mmr_vars),), dtype=np.float32
            )
            stats_grp.create_dataset(
                f"{var_type}_std", shape=(len(plast_mmr_vars),), dtype=np.float32
            )
            stats_grp.create_dataset(
                f"{var_type}_min", shape=(len(plast_mmr_vars),), dtype=np.float32
            )
            stats_grp.create_dataset(
                f"{var_type}_max", shape=(len(plast_mmr_vars),), dtype=np.float32
            )

        # Create datasets for meteorological statistics
        for var in met_vars:
            stats_grp.create_dataset(f"{var}_mean", shape=(), dtype=np.float32)
            stats_grp.create_dataset(f"{var}_std", shape=(), dtype=np.float32)
            stats_grp.create_dataset(f"{var}_min", shape=(), dtype=np.float32)
            stats_grp.create_dataset(f"{var}_max", shape=(), dtype=np.float32)

        # Process each file
        time_idx = 0

        # Arrays to collect statistics for normalization
        mmr_data = []
        emis_data = []
        dry_dep_data = []
        wet_dep_data = []
        met_data = {var: [] for var in met_vars}

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

                    # Store month and year
                    month_data[time_idx] = int(month)
                    year_data[time_idx] = year

                    # Store meteorological variables
                    for var in met_vars:
                        if var == "PS" or var == "TROPLEV":
                            data = nc.variables[var][0, :, :]
                            dyn_grp[var][time_idx, :, :] = data
                            met_data[var].append(data)
                        else:
                            data = nc.variables[var][0, :, :, :]
                            dyn_grp[var][time_idx, :, :, :] = data
                            met_data[var].append(data)

                    # Store microplastic variables
                    for i, var in enumerate(small_particles):
                        data = nc.variables[var][0, :, :, :]
                        small_grp[var][time_idx, :, :, :] = data
                        mmr_data.append(data.flatten())

                    for i, var in enumerate(large_particles):
                        data = nc.variables[var][0, :, :, :]
                        large_grp[var][time_idx, :, :, :] = data
                        mmr_data.append(data.flatten())

                    # Store emission variables
                    for i, var in enumerate(plast_emis_vars):
                        data = nc.variables[var][0, :, :]
                        emissions_grp[var][time_idx, :, :] = data
                        emis_data.append(data.flatten())

                    # Store deposition variables
                    for i, var in enumerate(plast_dry_dep_vars):
                        data = nc.variables[var][0, :, :]
                        dry_dep_grp[var][time_idx, :, :] = data
                        dry_dep_data.append(data.flatten())

                    for i, var in enumerate(plast_wet_dep_vars):
                        data = nc.variables[var][0, :, :]
                        wet_dep_grp[var][time_idx, :, :] = data
                        wet_dep_data.append(data.flatten())

                    # Calculate total deposition (dry + wet)
                    for i in range(len(plast_mmr_vars)):
                        dry_var = plast_dry_dep_vars[i]
                        wet_var = plast_wet_dep_vars[i]
                        deposition_grp["total_deposition"][time_idx, i, :, :] = (
                            nc.variables[dry_var][0, :, :]
                            + nc.variables[wet_var][0, :, :]
                        )

                # Calculate spatial derivatives for each variable
                # Horizontal gradients (simplified central difference)
                if time_idx == 0:  # Only need to create these datasets once
                    for var in small_particles + large_particles:
                        deriv_grp.create_dataset(
                            f"{var}_dx",
                            shape=(time_dim, len(lev), len(lat), len(lon)),
                            chunks=(1, len(lev), len(lat), len(lon)),
                            compression="gzip",
                            compression_opts=4,
                            dtype=np.float32,
                        )
                        deriv_grp.create_dataset(
                            f"{var}_dy",
                            shape=(time_dim, len(lev), len(lat), len(lon)),
                            chunks=(1, len(lev), len(lat), len(lon)),
                            compression="gzip",
                            compression_opts=4,
                            dtype=np.float32,
                        )
                        deriv_grp.create_dataset(
                            f"{var}_dz",
                            shape=(time_dim, len(lev), len(lat), len(lon)),
                            compression="gzip",
                            compression_opts=4,
                            dtype=np.float32,
                        )

                # Calculate horizontal gradients for microplastics
                for var in small_particles + large_particles:
                    data = None
                    if var in small_particles:
                        data = small_grp[var][time_idx]
                    else:
                        data = large_grp[var][time_idx]

                    # Calculate dx (longitude gradient)
                    dx = np.zeros_like(data)
                    dx[:, :, 1:-1] = (data[:, :, 2:] - data[:, :, :-2]) / 2
                    dx[:, :, 0] = data[:, :, 1] - data[:, :, 0]
                    dx[:, :, -1] = data[:, :, -1] - data[:, :, -2]
                    deriv_grp[f"{var}_dx"][time_idx] = dx

                    # Calculate dy (latitude gradient)
                    dy = np.zeros_like(data)
                    dy[:, 1:-1, :] = (data[:, 2:, :] - data[:, :-2, :]) / 2
                    dy[:, 0, :] = data[:, 1, :] - data[:, 0, :]
                    dy[:, -1, :] = data[:, -1, :] - data[:, -2, :]
                    deriv_grp[f"{var}_dy"][time_idx] = dy

                    # Calculate dz (vertical gradient)
                    dz = np.zeros_like(data)
                    dz[1:-1, :, :] = (data[2:, :, :] - data[:-2, :, :]) / 2
                    dz[0, :, :] = data[1, :, :] - data[0, :, :]
                    dz[-1, :, :] = data[-1, :, :] - data[-2, :, :]
                    deriv_grp[f"{var}_dz"][time_idx] = dz

                time_idx += 1

        # Store time information
        coords_grp.create_dataset("time", data=time_data)
        coords_grp.create_dataset("date", data=date_data)
        coords_grp.create_dataset("month", data=month_data)
        coords_grp.create_dataset("year", data=year_data)

        # Add settling velocities based on empirical data from the table
        # Create datasets for settling velocities
        small_grp.create_dataset(
            "settling_velocity",
            shape=(len(small_particles),),
            dtype=np.float32,
        )
        large_grp.create_dataset(
            "settling_velocity",
            shape=(len(large_particles),),
            dtype=np.float32,
        )
        # Also store sphere settling velocities for large particles
        large_grp.create_dataset(
            "sphere_settling_velocity",
            shape=(len(large_particles),),
            dtype=np.float32,
        )

        # Empirical settling velocities from the table (in cm/s)
        # Note from the paper: particle larger than 20μm tends to be fiber
        # Elongated particles - fiber, have lower settling velocities
        # They tested with spherical larger particles and result are very different
        sphere_settling = [0.00097, 0.0087, 0.097, 0.39, 3.61, 14.39]
        fiber_settling = [None, None, None, None, 2.7, 4.98]

        # Add metadata about settling velocities
        mp_grp.attrs["settling_velocity_units"] = "cm/s"
        mp_grp.attrs["settling_velocity_description"] = (
            "Terminal settling velocity for particles in air"
        )

        # Set values for small particles (use sphere values)
        for j, var in enumerate(small_particles):
            # Extract size from variable name
            size_key = var.split("_")[0]  # e.g., "Plast01"
            small_grp["settling_velocity"][j] = sphere_settling[j]
            small_grp[var].attrs["settling_velocity_cm_s"] = sphere_settling[j]
            small_grp[var].attrs["diameter_um"] = particle_sizes[size_key]
            small_grp[var].attrs["shape"] = "sphere"

        # Set values for large particles (use fiber values)
        for j, var in enumerate(large_particles):
            # Extract size from variable name
            size_key = var.split("_")[0]  # e.g., "Plast05"
            idx = j + len(small_particles)
            large_grp["settling_velocity"][j] = fiber_settling[idx]
            large_grp["sphere_settling_velocity"][j] = sphere_settling[idx]
            large_grp[var].attrs["settling_velocity_cm_s"] = fiber_settling[idx]
            large_grp[var].attrs["sphere_settling_velocity_cm_s"] = sphere_settling[idx]
            large_grp[var].attrs["diameter_um"] = particle_sizes[size_key]
            large_grp[var].attrs["shape"] = "fiber"

        # Create settling velocity parameters as in the table
        # Parameter a(D) and b(D) from table
        parameter_a = [7.1e-8, 1.1e-7, 9.7e-5, 1.7e-4, 2.8e-4, 3.9e-4]
        parameter_b = [6.9e-7, 1.2e-6, 1.8e-3, 3.6e-3, 5.5e-3, 6.4e-3]

        # Store these parameters as metadata
        deposition_params = mp_grp.create_group("deposition_parameters")
        deposition_params.create_dataset("parameter_a", data=parameter_a)
        deposition_params.create_dataset("parameter_b", data=parameter_b)

        # Add attributes to explain these parameters
        deposition_params.attrs["parameter_a_description"] = (
            "Deposition parameter a(D) for each particle size"
        )
        deposition_params.attrs["parameter_b_description"] = (
            "Deposition parameter b(D) for each particle size"
        )
        deposition_params.attrs["particle_diameters_um"] = list(particle_sizes.values())

        # Add Cunningham correction factors
        Cc_values = [1.326, 1.082, 1.032, 1.016, 1.008, 1.003]
        deposition_params.create_dataset("Cc", data=Cc_values)
        deposition_params.attrs["Cc_description"] = (
            "Cunningham correction factor for each particle size"
        )

        # Calculate statistics for normalization
        # For microplastic mass mixing ratios
        mmr_data = np.array(mmr_data)
        for i, var in enumerate(plast_mmr_vars):
            stats_grp["mmr_mean"][i] = np.mean(mmr_data[i :: len(plast_mmr_vars)])
            stats_grp["mmr_std"][i] = np.std(mmr_data[i :: len(plast_mmr_vars)])
            stats_grp["mmr_min"][i] = np.min(mmr_data[i :: len(plast_mmr_vars)])
            stats_grp["mmr_max"][i] = np.max(mmr_data[i :: len(plast_mmr_vars)])

        # For emissions
        emis_data = np.array(emis_data)
        for i, var in enumerate(plast_emis_vars):
            stats_grp["emis_mean"][i] = np.mean(emis_data[i :: len(plast_emis_vars)])
            stats_grp["emis_std"][i] = np.std(emis_data[i :: len(plast_emis_vars)])
            stats_grp["emis_min"][i] = np.min(emis_data[i :: len(plast_emis_vars)])
            stats_grp["emis_max"][i] = np.max(emis_data[i :: len(plast_emis_vars)])

        # For deposition (combining dry and wet)
        dep_data = np.array(dry_dep_data) + np.array(wet_dep_data)
        for i in range(len(plast_mmr_vars)):
            stats_grp["dep_mean"][i] = np.mean(dep_data[i :: len(plast_mmr_vars)])
            stats_grp["dep_std"][i] = np.std(dep_data[i :: len(plast_mmr_vars)])
            stats_grp["dep_min"][i] = np.min(dep_data[i :: len(plast_mmr_vars)])
            stats_grp["dep_max"][i] = np.max(dep_data[i :: len(plast_mmr_vars)])

        # For meteorological variables
        for var in met_vars:
            data = np.array(met_data[var])
            stats_grp[f"{var}_mean"][()] = np.mean(data)
            stats_grp[f"{var}_std"][()] = np.std(data)
            stats_grp[f"{var}_min"][()] = np.min(data)
            stats_grp[f"{var}_max"][()] = np.max(data)

        # Create normalized versions of the data
        # This is a simple min-max normalization, but you could use other methods
        for time_idx in range(time_dim):
            # Normalize microplastic variables
            for i, var in enumerate(small_particles):
                data = small_grp[var][time_idx]
                min_val = stats_grp["mmr_min"][i]
                max_val = stats_grp["mmr_max"][i]
                if max_val > min_val:
                    norm_data = (data - min_val) / (max_val - min_val)
                else:
                    norm_data = data - min_val
                norm_grp[var][time_idx] = norm_data

            for i, var in enumerate(large_particles):
                data = large_grp[var][time_idx]
                min_val = stats_grp["mmr_min"][i + len(small_particles)]
                max_val = stats_grp["mmr_max"][i + len(small_particles)]
                if max_val > min_val:
                    norm_data = (data - min_val) / (max_val - min_val)
                else:
                    norm_data = data - min_val
                norm_grp[var][time_idx] = norm_data

            # Normalize emission variables
            for i, var in enumerate(plast_emis_vars):
                data = emissions_grp[var][time_idx]
                min_val = stats_grp["emis_min"][i]
                max_val = stats_grp["emis_max"][i]
                if max_val > min_val:
                    norm_data = (data - min_val) / (max_val - min_val)
                else:
                    norm_data = data - min_val
                norm_grp[var][time_idx] = norm_data

            # Normalize meteorological variables
            for var in met_vars:
                if var == "PS" or var == "TROPLEV":
                    data = dyn_grp[var][time_idx]
                    min_val = stats_grp[f"{var}_min"][()]
                    max_val = stats_grp[f"{var}_max"][()]
                    if max_val > min_val:
                        norm_data = (data - min_val) / (max_val - min_val)
                    else:
                        norm_data = data - min_val
                    norm_grp[var][time_idx] = norm_data
                else:
                    data = dyn_grp[var][time_idx]
                    min_val = stats_grp[f"{var}_min"][()]
                    max_val = stats_grp[f"{var}_max"][()]
                    if max_val > min_val:
                        norm_data = (data - min_val) / (max_val - min_val)
                    else:
                        norm_data = data - min_val
                    norm_grp[var][time_idx] = norm_data

        # Create ML-ready feature combinations
        # Combined normalized features for each particle size
        for i, size_key in enumerate(
            ["Plast01", "Plast02", "Plast03", "Plast04", "Plast05", "Plast06"]
        ):
            ml_grp.create_dataset(
                f"{size_key}_features",
                shape=(time_dim, len(lat), len(lon), 7),  # 7 features
            )

    elapsed_time = time.time() - start_time
    print(f"Processing complete. Total time: {elapsed_time:.2f} seconds")
    print(f"Output file: {output_file}")


if __name__ == "__main__":
    create_dataset()
