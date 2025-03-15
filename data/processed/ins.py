import h5py
import numpy as np
import os
from pathlib import Path
import argparse

try:
    from src.utils import PATH

    DEFAULT_FILE = PATH.PROCESSED_DATA / "mpidata_v2.h5"
except ImportError:
    print("Warning: Could not import PATH from src.utils, using relative path")
    DEFAULT_FILE = Path("data/processed/mpidata_v2.h5")


def print_attrs(name, obj):
    """Print all attributes of an HDF5 object"""
    if len(obj.attrs) > 0:
        print(f"\nAttributes for {name}:")
        for key, val in obj.attrs.items():
            print(f"  {key}: {val}")


def print_dataset_info(name, obj):
    """Print information about a dataset"""
    if isinstance(obj, h5py.Dataset):
        shape_str = " × ".join([str(s) for s in obj.shape])
        size_mb = obj.size * obj.dtype.itemsize / (1024 * 1024)
        print(f"  {name}: {obj.dtype}, shape: {shape_str}, size: {size_mb:.2f} MB")

        # Print statistics for numerical datasets that aren't too large
        if (
            obj.dtype.kind in ["i", "f"] and obj.size < 1e7
        ):  # Only for reasonably sized numerical datasets
            try:
                data = obj[()]
                print(
                    f"    Range: [{np.min(data):.6g}, {np.max(data):.6g}], Mean: {np.mean(data):.6g}, Std: {np.std(data):.6g}"
                )
                # Check for NaNs or Infs
                if not np.isfinite(data).all():
                    nan_count = np.isnan(data).sum()
                    inf_count = np.isinf(data).sum()
                    print(
                        f"    Warning: Contains {nan_count} NaNs and {inf_count} Infs"
                    )
            except Exception as e:
                print(f"    Could not compute statistics: {e}")


def explore_group(group, indent=0):
    """Recursively explore and print information about a group and its contents"""
    # Print group attributes
    if indent == 0:
        print("\n" + "=" * 80)
        print(f"GROUP: {group.name}")
        print("=" * 80)
    else:
        print("\n" + "-" * 80)
        print(f"GROUP: {group.name}")
        print("-" * 80)

    # Print attributes
    print_attrs(group.name, group)

    # Print datasets directly in this group
    if any(isinstance(group[name], h5py.Dataset) for name in group):
        print(f"\nDatasets in {group.name}:")
        for name, obj in group.items():
            if isinstance(obj, h5py.Dataset):
                print_dataset_info(name, obj)

    # Recursively explore subgroups
    for name, obj in group.items():
        if isinstance(obj, h5py.Group) and obj.name != group.name:
            explore_group(obj, indent + 1)


def inspect_h5_file(file_path):
    """Inspect and print information about an HDF5 file"""
    try:
        with h5py.File(file_path, "r") as f:
            # Print file overview
            print("\n" + "=" * 80)
            print(f"HDF5 FILE: {file_path}")
            print("=" * 80)
            print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")

            # Print root attributes
            print_attrs("/", f)

            # Print top-level structure
            print("\nTop-level groups:")
            for name, obj in f.items():
                if isinstance(obj, h5py.Group):
                    print(f"  /{name}/")
                elif isinstance(obj, h5py.Dataset):
                    print(f"  /{name}")

            # Explore each top-level group
            for name, obj in f.items():
                if isinstance(obj, h5py.Group):
                    explore_group(obj)
                elif isinstance(obj, h5py.Dataset):
                    print("\n" + "=" * 80)
                    print(f"TOP-LEVEL DATASET: /{name}")
                    print("=" * 80)
                    print_dataset_info(name, obj)
                    print_attrs(name, obj)

            # Print summary of microplastic variables
            if "physics/microplastics" in f:
                print("\n" + "=" * 80)
                print("MICROPLASTIC VARIABLES SUMMARY")
                print("=" * 80)

                # Small particles
                if "physics/microplastics/small_particles" in f:
                    small_group = f["physics/microplastics/small_particles"]
                    print("\nSmall particles:")
                    for name in small_group:
                        if name != "settling_velocity" and isinstance(
                            small_group[name], h5py.Dataset
                        ):
                            print(f"  {name}")

                # Large particles
                if "physics/microplastics/large_particles" in f:
                    large_group = f["physics/microplastics/large_particles"]
                    print("\nLarge particles:")
                    for name in large_group:
                        if name != "settling_velocity" and isinstance(
                            large_group[name], h5py.Dataset
                        ):
                            print(f"  {name}")

                # Settling velocities
                if "physics/microplastics/small_particles/settling_velocity" in f:
                    sv = f["physics/microplastics/small_particles/settling_velocity"][
                        ()
                    ]
                    print("\nSmall particle settling velocities (m/s):")
                    print(
                        f"  Range: [{np.min(sv):.6g}, {np.max(sv):.6g}], Mean: {np.mean(sv):.6g}"
                    )

                if "physics/microplastics/large_particles/settling_velocity" in f:
                    sv = f["physics/microplastics/large_particles/settling_velocity"][
                        ()
                    ]
                    print("\nLarge particle settling velocities (m/s):")
                    print(
                        f"  Range: [{np.min(sv):.6g}, {np.max(sv):.6g}], Mean: {np.mean(sv):.6g}"
                    )

            # Print time information
            if "coordinates/time" in f:
                time = f["coordinates/time"][()]
                print("\n" + "=" * 80)
                print("TIME INFORMATION")
                print("=" * 80)
                print(f"Number of time steps: {len(time)}")

                if "coordinates/year" in f and "coordinates/month" in f:
                    years = f["coordinates/year"][()]
                    months = f["coordinates/month"][()]
                    print(
                        f"Time range: {years[0]}-{months[0]:02d} to {years[-1]}-{months[-1]:02d}"
                    )
                    print(f"Years covered: {np.unique(years)}")

            # Print grid information
            if "coordinates/lon" in f and "coordinates/lat" in f:
                lon = f["coordinates/lon"][()]
                lat = f["coordinates/lat"][()]
                print("\n" + "=" * 80)
                print("GRID INFORMATION")
                print("=" * 80)
                print(f"Grid dimensions: {len(lat)} latitude × {len(lon)} longitude")
                print(f"Latitude range: [{lat.min():.2f}, {lat.max():.2f}]")
                print(f"Longitude range: [{lon.min():.2f}, {lon.max():.2f}]")

                if "coordinates/lev" in f:
                    lev = f["coordinates/lev"][()]
                    print(f"Vertical levels: {len(lev)}")
                    print(f"Vertical range: [{lev.min():.2f}, {lev.max():.2f}]")

                if "coordinates/cell_area" in f:
                    cell_area = f["coordinates/cell_area"][()]
                    total_area = np.sum(cell_area)
                    print(f"Total grid area: {total_area:.2f} km²")

            # Print ML features information
            if "ml_features" in f:
                print("\n" + "=" * 80)
                print("MACHINE LEARNING FEATURES")
                print("=" * 80)

                if "ml_features/normalized" in f:
                    norm_group = f["ml_features/normalized"]
                    print("\nNormalized variables available for ML:")
                    for name in norm_group:
                        print(f"  {name}")

                if "ml_features/statistics" in f:
                    stats_group = f["ml_features/statistics"]
                    print("\nStatistics available for normalization:")
                    for name in stats_group:
                        print(f"  {name}")

    except Exception as e:
        print(f"Error inspecting HDF5 file: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect an HDF5 file structure and contents"
    )
    parser.add_argument(
        "--file", type=str, default=None, help="Path to the HDF5 file to inspect"
    )
    args = parser.parse_args()

    file_path = args.file if args.file else DEFAULT_FILE

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return

    inspect_h5_file(file_path)


if __name__ == "__main__":
    main()
