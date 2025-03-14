# %% Imports
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

from src.utils import PATH
from src.utils.data.loader import MPIDataLoader, MPIDataset

# %% Load Data
loader = MPIDataLoader(PATH.PROCESSED_DATA / "mpidata.h5")
data = loader.load_full_dataset()

# Convert coordinates to numpy arrays
lat_array = np.array(data.lat)
lon_array = np.array(data.lon)
time_array = np.array(data.time)
date_array = np.array(data.date)


# %% Hemispheric Emission Analysis
def plot_hemispheric_emissions():
    """Compare emissions between Northern and Southern hemispheres"""
    emission_var = list(data.emissions.keys())[0]
    emissions = data.emissions[emission_var]

    # Split hemispheres
    nh_mask = lat_array > 0
    sh_mask = lat_array <= 0

    # Calculate total emissions per hemisphere over time
    nh_emissions = np.sum(emissions[:, nh_mask, :], axis=(1, 2))
    sh_emissions = np.sum(emissions[:, sh_mask, :], axis=(1, 2))

    dates = [datetime.strptime(str(d), "%Y%m%d") for d in date_array]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=nh_emissions, name="Northern Hemisphere"))
    fig.add_trace(go.Scatter(x=dates, y=sh_emissions, name="Southern Hemisphere"))

    fig.update_layout(
        title="Hemispheric Comparison of Microplastic Emissions",
        xaxis_title="Date",
        yaxis_title="Total Emissions",
        height=600,
        width=1000,
    )

    # Calculate and print emission statistics
    nh_mean = np.mean(nh_emissions)
    sh_mean = np.mean(sh_emissions)
    ratio = nh_mean / sh_mean
    print(f"NH/SH Emission Ratio: {ratio:.2f}")

    return fig


fig_emissions = plot_hemispheric_emissions()
fig_emissions.show()


# %% Wind Patterns and Transport Analysis
def plot_wind_transport_patterns():
    """Analyze wind patterns and their influence on particle transport"""
    # Get seasonal averages of wind components
    u_wind = np.array(data.dynamics["U"])
    v_wind = np.array(data.dynamics["V"])

    # Create seasonal masks (assuming data starts in January)
    months = [datetime.strptime(str(d), "%Y%m%d").month for d in date_array]
    seasons = {
        "DJF": [12, 1, 2],
        "MAM": [3, 4, 5],
        "JJA": [6, 7, 8],
        "SON": [9, 10, 11],
    }

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=list(seasons.keys()),
        shared_xaxes=True,
        shared_yaxes=True,
    )

    # Reduce density of wind vectors for clarity
    skip = 4  # Plot every 4th point
    lon_subset = lon_array[::skip]
    lat_subset = lat_array[::skip]

    # Create meshgrid for quiver plot
    lon_mesh, lat_mesh = np.meshgrid(lon_subset, lat_subset)

    for i, (season, months_in_season) in enumerate(seasons.items()):
        season_mask = [m in months_in_season for m in months]

        # Average wind components for the season
        u_seasonal = np.mean(u_wind[season_mask, 0], axis=0)[
            ::skip, ::skip
        ]  # surface level
        v_seasonal = np.mean(v_wind[season_mask, 0], axis=0)[::skip, ::skip]

        # Calculate wind speed for color scaling
        wind_speed = np.sqrt(u_seasonal**2 + v_seasonal**2)

        row = (i // 2) + 1
        col = (i % 2) + 1

        # Plot wind vectors as arrows
        fig.add_trace(
            go.Scatter(
                x=lon_mesh.flatten(),
                y=lat_mesh.flatten(),
                mode="markers",
                marker=dict(
                    symbol="arrow",
                    angle=np.arctan2(v_seasonal, u_seasonal).flatten() * 180 / np.pi
                    - 90,
                    size=5,
                    color=wind_speed.flatten(),
                    colorscale="Viridis",
                    showscale=True if i == 0 else False,  # Show colorbar only once
                    colorbar=dict(title="Wind Speed (m/s)") if i == 0 else None,
                ),
                name=season,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        # Update subplot axes
        fig.update_xaxes(title_text="Longitude" if row == 2 else None, row=row, col=col)
        fig.update_yaxes(title_text="Latitude" if col == 1 else None, row=row, col=col)

    fig.update_layout(
        title="Seasonal Wind Patterns and Transport",
        height=800,
        width=1200,
    )

    return fig


fig_wind = plot_wind_transport_patterns()
fig_wind.show()


# %% Hemispheric Deposition Patterns
def plot_hemispheric_deposition():
    """Compare deposition patterns between hemispheres"""
    dry_var = list(data.dry_deposition.keys())[0]
    wet_var = list(data.wet_deposition.keys())[0]

    # Split into hemispheres
    nh_mask = lat_array > 0
    sh_mask = lat_array <= 0

    # Calculate zonal means for each hemisphere
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Northern Hemisphere", "Southern Hemisphere")
    )

    # Process each hemisphere
    for hemisphere, mask, col in [("NH", nh_mask, 1), ("SH", sh_mask, 2)]:
        dry_dep = np.array(data.dry_deposition[dry_var])
        wet_dep = np.array(data.wet_deposition[wet_var])

        # Calculate total deposition and its temporal evolution
        total_dep = dry_dep + wet_dep
        zonal_mean = np.mean(total_dep[:, mask, :], axis=2)

        fig.add_trace(
            go.Heatmap(
                z=zonal_mean.T,
                x=time_array,
                y=lat_array[mask],
                colorscale="Viridis",
                colorbar=dict(title="Deposition Rate"),
            ),
            row=1,
            col=col,
        )

    fig.update_layout(
        title="Hemispheric Comparison of Deposition Patterns", height=600, width=1200
    )

    return fig


fig_deposition = plot_hemispheric_deposition()
fig_deposition.show()


# %% Latitudinal Transport Analysis
def analyze_latitudinal_transport():
    """Analyze particle transport across different latitude bands"""
    small_var = list(data.small_particles.keys())[0]
    particles = np.array(data.small_particles[small_var])

    # Define latitude bands
    lat_bands = [(-90, -60), (-60, -30), (-30, 0), (0, 30), (30, 60), (60, 90)]

    # Calculate concentration evolution in each band
    fig = go.Figure()

    for lat_min, lat_max in lat_bands:
        mask = (lat_array >= lat_min) & (lat_array < lat_max)
        band_conc = np.mean(particles[:, :, mask, :], axis=(1, 2, 3))

        fig.add_trace(
            go.Scatter(x=time_array, y=band_conc, name=f"{lat_min}° to {lat_max}°")
        )

    fig.update_layout(
        title="Particle Concentration Evolution by Latitude Band",
        xaxis_title="Time",
        yaxis_title="Mean Concentration",
        height=600,
        width=1000,
    )

    return fig


fig_transport = analyze_latitudinal_transport()
fig_transport.show()
