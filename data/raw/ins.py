# %%
from src.utils import PATH
import os
from netCDF4 import Dataset

# %% Get file information
sample = PATH.RAW_DATA / "mpidata" / "MZ4_plastic_6sp_wd_oldsettl.mz4.h0.2013-01.nc"

print("File Information:")
if os.path.exists(sample):
    print(f"- File path: {sample}")
    print(f"- File size: {os.path.getsize(sample) / (1024*1024):.2f} MB")
    print(f"- Last modified: {os.path.getmtime(sample)}")
else:
    print(f"File not found: {sample}")

# %%
with Dataset(sample, "r") as nc:
    print("\nNetCDF File Structure:")
    print("======================")

    # Print dimensions
    print("\nDimensions:")
    for dim_name, dimension in nc.dimensions.items():
        print(f"- {dim_name}: {len(dimension)}")

    # Print variables
    print("\nVariables:")
    for var_name, variable in nc.variables.items():
        print(f"\n{var_name}:")
        print(f"  Shape: {variable.shape}")
        print(f"  Data type: {variable.dtype}")
        if hasattr(variable, "units"):
            print(f"  Units: {variable.units}")
        if hasattr(variable, "long_name"):
            print(f"  Long name: {variable.long_name}")

    # Print global attributes
    print("\nGlobal Attributes:")
    for attr_name in nc.ncattrs():
        print(f"- {attr_name}: {getattr(nc, attr_name)}")
