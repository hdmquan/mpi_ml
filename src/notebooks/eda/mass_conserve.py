# %% [markdown]
# # Mass Conservation Analysis
# This notebook analyzes the temporal evolution of wet deposition, dry deposition, and mass mixing ratio (MMR) for different particle sizes.

# %% Imports
import h5py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.utils import PATH

# %% Load and process data
input_file = PATH.PROCESSED_DATA / "mpidata.h5"

results = []

with h5py.File(input_file, "r") as hf:
    # Get timestamps
    timestamps = [ts.decode() for ts in hf["metadata/time"][:]]

    # Process each particle size (1-6)
    for i in range(1, 7):
        particle_results = []

        for month_idx in range(len(timestamps)):
            # Get wet deposition
            wet_dep = hf[f"outputs/deposition/Plast0{i}_WETDEP_FLUX_avrg"][month_idx]
            # Get dry deposition
            dry_dep = hf[f"outputs/deposition/Plast0{i}_DRY_DEP_FLX_avrg"][month_idx]
            # Get MMR (sum across all levels)
            mmr = hf[f"outputs/mass_mixing_ratio/Plast0{i}_MMR_avrg"][month_idx]

            # Calculate sums
            wet_sum = np.sum(wet_dep)
            dry_sum = np.sum(dry_dep)
            mmr_sum = np.sum(mmr)

            particle_results.append(
                {
                    "timestamp": timestamps[month_idx],
                    "particle_size": f"Size {i}",
                    "wet_deposition": wet_sum,
                    "dry_deposition": dry_sum,
                    "mmr": mmr_sum,
                    "total": wet_sum + dry_sum + mmr_sum,
                }
            )

        results.extend(particle_results)

# Convert to DataFrame
df = pd.DataFrame(results)

# %% [markdown]
# ## Time Series Visualization of Components by Particle Size

# %% Create interactive time series plot
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "Wet Deposition",
        "Dry Deposition",
        "Mass Mixing Ratio",
        "Total (Wet + Dry + MMR)",
    ),
    vertical_spacing=0.12,
    horizontal_spacing=0.1,
)

components = {
    "wet_deposition": (1, 1),
    "dry_deposition": (1, 2),
    "mmr": (2, 1),
    "total": (2, 2),
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
    height=800,
    width=1200,
    title_text="Temporal Evolution of Deposition Components by Particle Size",
    showlegend=True,
)

fig.update_xaxes(title_text="Time")
fig.update_yaxes(title_text="Value")

fig.show()

# %% [markdown]
# ## Stacked Area Chart of Total Components

# %% Create stacked area chart
df_pivot = df.pivot_table(
    index="timestamp", columns="particle_size", values="total", aggfunc="sum"
).reset_index()

fig_area = go.Figure()

for size in df["particle_size"].unique():
    fig_area.add_trace(
        go.Scatter(
            x=df_pivot["timestamp"],
            y=df_pivot[size],
            name=size,
            stackgroup="one",
            mode="lines",
            line=dict(width=0.5),
        )
    )

fig_area.update_layout(
    title="Stacked Total Components Over Time",
    xaxis_title="Time",
    yaxis_title="Total Value",
    height=500,
    width=1000,
    showlegend=True,
)

fig_area.show()
