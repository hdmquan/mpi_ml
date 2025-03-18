import h5py
import os
from src.utils import PATH


def explore_h5_structure():
    """
    Print out the structure and dimensions of the HDF5 file created by the data processing script.
    This provides a simple overview for users unfamiliar with the data format.
    """

    h5_file_path = PATH.PROCESSED_DATA / "mpidata.h5"

    if not os.path.exists(h5_file_path):
        print(f"Error: File {h5_file_path} not found.")
        print("Please run the process_netcdf_files() function first.")
        return

    print(f"\n{'='*80}")
    print(f"EXPLORING HDF5 FILE: {h5_file_path}")
    print(f"{'='*80}\n")

    with h5py.File(h5_file_path, "r") as h5f:

        print("FILE ATTRIBUTES:")
        print("--------------")
        for attr_name, attr_value in h5f.attrs.items():
            print(f"{attr_name}: {attr_value}")
        print()

        def explore_group(group, indent=0):
            for key in group.keys():
                item = group[key]
                spacing = "  " * indent

                if isinstance(item, h5py.Group):

                    print(f"{spacing}GROUP: {key}/")
                    explore_group(item, indent + 1)
                else:

                    shape_str = " × ".join([str(dim) for dim in item.shape])
                    print(f"{spacing}DATASET: {key} - Shape: {shape_str}")

                    if len(item.attrs) > 0:
                        print(f"{spacing}  Attributes: ", end="")
                        attrs = list(item.attrs.keys())
                        if len(attrs) > 3:
                            print(f"{', '.join(attrs[:3])}, ... ({len(attrs)} total)")
                        else:
                            print(f"{', '.join(attrs)}")

        print("FILE STRUCTURE:")
        print("--------------")
        explore_group(h5f)

        print("\nSUMMARY:")
        print("-------")

        time_dim = h5f["metadata"]["time"].shape[0]
        print(f"Time periods: {time_dim} months")

        lats = h5f["metadata"]["lat"][:]
        lons = h5f["metadata"]["lon"][:]
        levs = h5f["metadata"]["lev"][:]
        print(
            f"Spatial dimensions: {len(lats)} latitudes × {len(lons)} longitudes × {len(levs)} vertical levels"
        )
        print(f"Settling velocities (m/s): {h5f['metadata']['settling_velocity'][:]}")

        print(f"\nInput variables (meteorological data): {len(h5f['inputs'].keys())}")
        print(f"Output variables:")
        print(
            f"  - Mass mixing ratio variables: {len(h5f['outputs']['mass_mixing_ratio'].keys())}"
        )
        print(
            f"  - Deposition flux variables: {len(h5f['outputs']['deposition'].keys())}"
        )


if __name__ == "__main__":
    explore_h5_structure()
