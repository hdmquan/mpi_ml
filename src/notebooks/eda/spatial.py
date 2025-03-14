# %% Imports
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from src.utils import PATH
from src.utils.data.loader import MPIDataLoader, MPIDataset

# %% Load Data
loader = MPIDataLoader(PATH.PROCESSED_DATA / "mpidata.h5")
data = loader.load_full_dataset()


# %% Global Distribution Map - Surface Level
def plot_global_distribution(
    var_data: np.ndarray, lat: np.ndarray, lon: np.ndarray, title: str
):
    # Normalize data to improve visibility
    # Add small epsilon to avoid division by zero
    normalized_data = (var_data - np.min(var_data)) / (
        np.max(var_data) - np.min(var_data) + 1e-10
    )

    return px.imshow(
        normalized_data,
        x=lon,
        y=lat,
        origin="lower",
        aspect="equal",
        title=title,
        labels={"x": "Longitude", "y": "Latitude", "color": "Concentration"},
        color_continuous_scale="Viridis",
        contrast_rescaling="minmax",  # Helps with contrast
    ).update_layout(
        template="plotly_dark",
        width=1000,
        height=600,
        coloraxis_colorbar=dict(title="Normalized Concentration"),
    )


# Plot small particles surface concentration (first size bin, first time step)
small_particle_surface = data.small_particles[list(data.small_particles.keys())[0]][
    0, 0
]
fig_surface = plot_global_distribution(
    small_particle_surface, data.lat, data.lon, "Surface Distribution - Small Particles"
)
fig_surface.show()


# %% Vertical Profile Analysis
def plot_vertical_profile():
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Small Particles", "Large Particles")
    )

    # Average over time and longitude for zonal mean
    small_conc = data.small_particles[list(data.small_particles.keys())[0]]
    large_conc = data.large_particles[list(data.large_particles.keys())[0]]

    zonal_small = np.mean(small_conc, axis=(0, 3))  # average over time and longitude
    zonal_large = np.mean(large_conc, axis=(0, 3))

    fig.add_trace(
        go.Heatmap(z=zonal_small, y=data.lev, x=data.lat, colorscale="Viridis"),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(z=zonal_large, y=data.lev, x=data.lat, colorscale="Viridis"),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=600,
        width=1200,
        title_text="Zonal Mean Vertical Distribution",
        yaxis_title="Pressure Level (hPa)",
        yaxis_type="log",
        yaxis2_type="log",
    )

    return fig


fig_vertical = plot_vertical_profile()
fig_vertical.show()


# %% Emission and Deposition Patterns
def plot_emission_deposition():
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Surface Emissions",
            "Dry Deposition",
            "Wet Deposition",
            "Net Deposition",
        ),
    )

    # Get first variable from each category
    emissions = np.mean(data.emissions[list(data.emissions.keys())[0]], axis=0)
    dry_dep = np.mean(data.dry_deposition[list(data.dry_deposition.keys())[0]], axis=0)
    wet_dep = np.mean(data.wet_deposition[list(data.wet_deposition.keys())[0]], axis=0)
    net_dep = dry_dep + wet_dep

    plots = [(emissions, 1, 1), (dry_dep, 1, 2), (wet_dep, 2, 1), (net_dep, 2, 2)]

    for plot_data, row, col in plots:
        fig.add_trace(
            go.Heatmap(z=plot_data, y=data.lat, x=data.lon, colorscale="Viridis"),
            row=row,
            col=col,
        )

    fig.update_layout(
        height=1000, width=1200, title_text="Emission and Deposition Patterns"
    )

    return fig


fig_deposition = plot_emission_deposition()
fig_deposition.show()


# %% Transport Analysis
def plot_transport_analysis():
    # Get wind components and particle concentration
    u_wind = np.mean(data.dynamics["U"], axis=0)[0]  # surface level
    v_wind = np.mean(data.dynamics["V"], axis=0)[0]
    particles = np.mean(
        data.small_particles[list(data.small_particles.keys())[0]], axis=0
    )[0]

    fig = go.Figure()

    # Add particle concentration as contour
    fig.add_trace(
        go.Contour(
            z=particles,
            x=data.lon,
            y=data.lat,
            colorscale="Viridis",
            name="Particle Concentration",
        )
    )

    # Add wind vectors
    skip = 3  # Plot every nth point to avoid overcrowding
    fig.add_trace(
        go.Scatter(
            x=data.lon[::skip],
            y=data.lat[::skip],
            mode="markers+text",
            marker=dict(
                symbol="arrow",
                angle=np.arctan2(v_wind[::skip, ::skip], u_wind[::skip, ::skip])
                * 180
                / np.pi,
                size=8,
            ),
            name="Wind Vectors",
        )
    )

    fig.update_layout(
        height=800, width=1200, title_text="Transport Patterns with Wind Vectors"
    )

    return fig


fig_transport = plot_transport_analysis()
fig_transport.show()


# %% Time Evolution at Selected Points
def plot_time_evolution():
    # Select a few representative points
    lats = [0, 30, 60]  # equator, mid-latitude, high-latitude
    small_particle_var = list(data.small_particles.keys())[0]

    fig = go.Figure()

    for lat_idx in [np.abs(data.lat - lat).argmin() for lat in lats]:
        concentration = data.small_particles[small_particle_var][:, 0, lat_idx, :]
        mean_conc = np.mean(concentration, axis=1)

        fig.add_trace(
            go.Scatter(
                x=data.time, y=mean_conc, name=f"Latitude {data.lat[lat_idx]:.1f}°"
            )
        )

    fig.update_layout(
        height=600,
        width=1000,
        title_text="Time Evolution at Different Latitudes",
        xaxis_title="Time",
        yaxis_title="Concentration",
    )

    return fig


fig_time = plot_time_evolution()
fig_time.show()
