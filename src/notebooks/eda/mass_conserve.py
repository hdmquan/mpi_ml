# %% Imports
import h5py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
import datetime
from src.utils import PATH

# %% Load and process data
input_file = PATH.PROCESSED_DATA / "mpidata.h5"

# Load data from HDF5 file
results = []
with h5py.File(input_file, "r") as hf:
    timestamps = [ts.decode() for ts in hf["metadata/time"][:]]

    for i in range(1, 7):  # Process each particle size (1-6)
        for month_idx in range(len(timestamps)):
            wet_dep = hf[f"outputs/deposition/Plast0{i}_WETDEP_FLUX_avrg"][month_idx]
            dry_dep = hf[f"outputs/deposition/Plast0{i}_DRY_DEP_FLX_avrg"][month_idx]
            mmr = hf[f"outputs/mass_mixing_ratio/Plast0{i}_MMR_avrg"][month_idx]
            surf_emis = hf[f"outputs/emissions/Plast0{i}_SRF_EMIS_avrg"][month_idx]

            results.append(
                {
                    "timestamp": timestamps[month_idx],
                    "particle_size": f"Size {i}",
                    "wet_deposition": np.sum(wet_dep),
                    "dry_deposition": np.sum(dry_dep),
                    "mmr": np.sum(mmr),
                    "surface_emission": np.sum(surf_emis),
                }
            )

df = pd.DataFrame(results)
df["total"] = (
    df["wet_deposition"] + df["dry_deposition"] + df["mmr"] + df["surface_emission"]
)

# %% [markdown]
# ## Time Series Visualization of Components by Particle Size

# %% Create interactive time series plot
# Create multi-panel time series plot of all components
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "Wet Deposition",
        "Dry Deposition",
        "Mass Mixing Ratio",
        "Surface Emission",
    ),
    vertical_spacing=0.12,
    horizontal_spacing=0.1,
)

components = {
    "wet_deposition": (1, 1),
    "dry_deposition": (1, 2),
    "mmr": (2, 1),
    "surface_emission": (2, 2),
}

for component, (row, col) in components.items():
    for size in df["particle_size"].unique():
        mask = df["particle_size"] == size
        fig.add_trace(
            go.Scatter(
                x=df[mask]["timestamp"],
                y=df[mask][component],
                name=f"{size} - {component}",
                showlegend=True if col == 2 else False,
            ),
            row=row,
            col=col,
        )

fig.update_layout(
    height=1000,
    width=1200,
    title_text="Temporal Evolution of Deposition Components by Particle Size",
    showlegend=True,
)
fig.update_xaxes(title_text="Time")
fig.update_yaxes(title_text="Value")

fig.show()

# So it seems Surface Emission is constant like they stated in the paper

# Source unit is kg/m2
# MMR unit is kg/kg/cell_volume
# Deposition unit is kg/cell_area
# To able to make mass conservation check we need to convert everything to kg
# Formula is MMR_sum * accounted_atmosphere_weight = source_avg * earth_surface_area + wet_deposition_sum + dry_deposition_sum
# source_avg should be calculated carefully as cell area is not constant
# Hence accounted_atmosphere_weight = (source_sum * earth_surface_area + wet_deposition_sum + dry_deposition_sum) / MMR_sum

# %% Constants and Calculations
EARTH_RADIUS = 6.371e6  # Earth's radius in meters
EARTH_SURFACE_AREA = 4 * np.pi * EARTH_RADIUS**2  # Earth's surface area in m²

# Calculate accounted atmosphere weight for each particle and timestamp
df["source_mass"] = df["surface_emission"] * EARTH_SURFACE_AREA
weights = []

for size in df["particle_size"].unique():
    size_data = df[df["particle_size"] == size]
    for _, row in size_data.iterrows():
        total_mass = row["source_mass"] + row["wet_deposition"] + row["dry_deposition"]
        acc_weight = total_mass / row["mmr"] if row["mmr"] != 0 else np.nan
        weights.append(
            {
                "particle_size": size,
                "timestamp": row["timestamp"],
                "accounted_weight": acc_weight,
            }
        )

acc_weight_df = pd.DataFrame(weights)

# %% Visualize accounted atmosphere weight over time
fig = px.line(
    acc_weight_df,
    x="timestamp",
    y="accounted_weight",
    color="particle_size",
    title="Accounted Atmosphere Weight Over Time",
)
fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Accounted Atmosphere Weight (kg)",
    height=600,
    width=1000,
)
fig.show()

# %% Calculate and visualize mass conservation ratio
# Calculate mass conservation ratio for each particle and timestamp
conservation_results = []

for size in df["particle_size"].unique():
    size_data = df[df["particle_size"] == size]
    size_weight_df = acc_weight_df[acc_weight_df["particle_size"] == size]

    for _, row in size_data.iterrows():
        acc_weight = size_weight_df[size_weight_df["timestamp"] == row["timestamp"]][
            "accounted_weight"
        ].values[0]

        theoretical_mass = (
            row["mmr"] * acc_weight if not np.isnan(acc_weight) else np.nan
        )
        actual_mass = row["source_mass"] + row["wet_deposition"] + row["dry_deposition"]
        conservation_ratio = (
            theoretical_mass / actual_mass if actual_mass != 0 else np.nan
        )

        conservation_results.append(
            {
                "particle_size": size,
                "timestamp": row["timestamp"],
                "conservation_ratio": conservation_ratio,
            }
        )

conservation_df = pd.DataFrame(conservation_results)

# Plot mass conservation ratio over time
fig = px.line(
    conservation_df,
    x="timestamp",
    y="conservation_ratio",
    color="particle_size",
    title="Mass Conservation Ratio Over Time",
)
fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Mass Conservation Ratio",
    height=600,
    width=1000,
    yaxis=dict(range=[0, 2]),
)
fig.add_hline(
    y=1.0,
    line_dash="dash",
    line_color="red",
    annotation_text="Perfect Conservation",
)
fig.show()

# The graph seems to be 1 cycle of cosine over a year
# Check concentration ratio between the 2 hemispheres...


# %% Fit cosine model to accounted atmosphere weight
# Define cosine model function
def cosine_model(t, amplitude, phase, vertical_shift, period=365.25):
    """Cosine function with amplitude, phase shift, vertical shift and period in days"""
    return amplitude * np.cos(2 * np.pi * t / period + phase) + vertical_shift


# Convert timestamp string to days since first timestamp
def timestamp_to_days(timestamp_str, first_timestamp_str):
    """Convert timestamp string to days since first timestamp"""
    timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m")
    first_timestamp = datetime.datetime.strptime(first_timestamp_str, "%Y-%m")
    return (timestamp - first_timestamp).days


# Fit cosine model to accounted atmosphere weight for each particle size
cosine_params = {}

for size in acc_weight_df["particle_size"].unique():
    size_data = acc_weight_df[acc_weight_df["particle_size"] == size]

    timestamps_list = size_data["timestamp"].tolist()
    first_timestamp = timestamps_list[0]
    days = [timestamp_to_days(ts, first_timestamp) for ts in timestamps_list]
    weights = size_data["accounted_weight"].values

    if np.isnan(weights).any():
        print(f"Skipping {size} due to NaN values")
        continue

    # Initial parameter guesses
    P0 = [
        (np.max(weights) - np.min(weights)) / 2,  # amplitude
        0,  # phase
        np.mean(weights),  # vertical shift
    ]

    try:
        params, _ = curve_fit(cosine_model, days, weights, P0=P0)
        cosine_params[size] = {
            "amplitude": params[0],
            "phase": params[1],
            "vertical_shift": params[2],
        }

        # Plot fitted cosine curve against data points
        days_fine = np.linspace(min(days), max(days), 1000)
        fitted_weights = cosine_model(days_fine, *params)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=days, y=weights, mode="markers", name=f"{size} Data")
        )
        fig.add_trace(
            go.Scatter(
                x=days_fine,
                y=fitted_weights,
                mode="lines",
                name="Fitted Cosine",
                line=dict(color="red"),
            )
        )
        fig.update_layout(
            title=f"Cosine Fit for {size}",
            xaxis_title=f"Days since {first_timestamp}",
            yaxis_title="Accounted Atmosphere Weight (kg)",
            height=600,
            width=1000,
            showlegend=True,
        )
        fig.show()

        # Print cosine model parameters and fit quality
        print(f"Cosine parameters for {size}:")
        print(f"  Amplitude: {params[0]:.2e}")
        print(f"  Phase: {params[1]:.4f}")
        print(f"  Vertical shift: {params[2]:.2e}")
        print(
            f"  R² value: {np.corrcoef(weights, cosine_model(np.array(days), *params))[0, 1]**2:.4f}"
        )

    except Exception as e:
        print(f"Could not fit cosine model for {size}: {e}")

# %%
