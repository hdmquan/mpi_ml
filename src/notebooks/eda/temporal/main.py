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

# Convert coordinates to numpy arrays for easier manipulation
lat_array = np.array(data.lat)
lon_array = np.array(data.lon)
time_array = np.array(data.time)
date_array = np.array(data.date)


# %% Time Series Analysis
def plot_global_mean_timeseries():
    """Plot global mean concentrations over time for both particle sizes"""
    fig = make_subplots(rows=1, cols=1)

    # Calculate global means for both particle sizes
    small_var = list(data.small_particles.keys())[0]
    large_var = list(data.large_particles.keys())[0]

    small_global = np.mean(data.small_particles[small_var], axis=(1, 2, 3))
    large_global = np.mean(data.large_particles[large_var], axis=(1, 2, 3))

    # Convert dates to datetime
    dates = [datetime.strptime(str(d), "%Y%m%d") for d in date_array]

    fig.add_trace(go.Scatter(x=dates, y=small_global, name="Small Particles"))
    fig.add_trace(go.Scatter(x=dates, y=large_global, name="Large Particles"))

    fig.update_layout(
        title="Global Mean Particle Concentrations Over Time",
        xaxis_title="Date",
        yaxis_title="Concentration",
        height=600,
        width=1000,
    )

    return fig


fig_timeseries = plot_global_mean_timeseries()
fig_timeseries.show()


# %% Seasonal Patterns
def plot_seasonal_patterns():
    """Analyze seasonal patterns in particle concentrations"""
    small_var = list(data.small_particles.keys())[0]
    dates = [datetime.strptime(str(d), "%Y%m%d") for d in date_array]
    months = [d.month for d in dates]

    # Calculate monthly averages for different latitudinal bands
    lat_bands = [
        (-90, -60, "High South"),
        (-60, -30, "Mid South"),
        (-30, 30, "Tropics"),
        (30, 60, "Mid North"),
        (60, 90, "High North"),
    ]

    fig = go.Figure()

    for lat_min, lat_max, band_name in lat_bands:
        lat_mask = (lat_array >= lat_min) & (lat_array < lat_max)
        band_data = data.small_particles[small_var][:, 0, lat_mask, :]
        monthly_means = [
            np.mean(band_data[np.array(months) == m]) for m in range(1, 13)
        ]

        fig.add_trace(
            go.Scatter(
                x=list(range(1, 13)),
                y=monthly_means,
                name=band_name,
                mode="lines+markers",
            )
        )

    fig.update_layout(
        title="Seasonal Patterns by Latitudinal Bands",
        xaxis_title="Month",
        yaxis_title="Mean Concentration",
        xaxis=dict(
            tickmode="array",
            ticktext=[
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ],
            tickvals=list(range(1, 13)),
        ),
        height=600,
        width=1000,
    )

    return fig


fig_seasonal = plot_seasonal_patterns()
fig_seasonal.show()


# %% Deposition Time Evolution
def plot_deposition_evolution():
    """Analyze temporal evolution of deposition patterns"""
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Dry Deposition", "Wet Deposition")
    )

    dry_var = list(data.dry_deposition.keys())[0]
    wet_var = list(data.wet_deposition.keys())[0]

    # Calculate zonal means over time
    dry_zonal = np.mean(data.dry_deposition[dry_var], axis=2)  # average over longitude
    wet_zonal = np.mean(data.wet_deposition[wet_var], axis=2)

    # Create Hovmöller diagrams
    fig.add_trace(
        go.Heatmap(
            z=dry_zonal.T,
            x=time_array,
            y=lat_array,
            colorscale="Viridis",
            colorbar=dict(title="Dry Deposition Rate"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=wet_zonal.T,
            x=time_array,
            y=lat_array,
            colorscale="Viridis",
            colorbar=dict(title="Wet Deposition Rate"),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=500,
        width=1200,
        title_text="Temporal Evolution of Deposition Patterns",
        xaxis_title="Time",
        yaxis_title="Latitude",
    )

    return fig


fig_deposition = plot_deposition_evolution()
fig_deposition.show()


# %% Transport Variability
def plot_transport_variability():
    """Analyze temporal variability in transport patterns"""
    # Calculate wind speed and its temporal variation
    u_wind = data.dynamics["U"][:, 0]  # surface level
    v_wind = data.dynamics["V"][:, 0]

    wind_speed = np.sqrt(u_wind**2 + v_wind**2)
    wind_variability = np.std(wind_speed, axis=0)

    fig = go.Figure(
        data=go.Contour(
            z=wind_variability,
            x=lon_array,
            y=lat_array,
            colorscale="Viridis",
            colorbar=dict(title="Wind Speed Variability (m/s)"),
        )
    )

    fig.update_layout(
        title="Temporal Variability in Wind Patterns",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=600,
        width=1000,
    )

    return fig


fig_transport = plot_transport_variability()
fig_transport.show()

# %%
