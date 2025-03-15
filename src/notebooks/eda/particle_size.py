# %% Imports
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

from src.utils import PATH
from src.data.loader import MPIDataLoader, MPIDataset

# %% Load Data
loader = MPIDataLoader(PATH.PROCESSED_DATA / "mpidata.h5")
data = loader.load_full_dataset()

# Convert coordinates to numpy arrays
lat_array = np.array(data.lat)
lon_array = np.array(data.lon)
lev_array = np.array(data.lev)


# %% Debug Data Structure
def print_data_structure():
    """Print the structure of the data to understand its organization"""
    print("Small Particles Variables:", list(data.small_particles.keys()))
    print("Large Particles Variables:", list(data.large_particles.keys()))

    # Print shape of first variable in each category
    small_var = list(data.small_particles.keys())[0]
    large_var = list(data.large_particles.keys())[0]

    print("\nData Shapes:")
    print(
        f"Small particles ({small_var}):",
        np.array(data.small_particles[small_var]).shape,
    )
    print(
        f"Large particles ({large_var}):",
        np.array(data.large_particles[large_var]).shape,
    )
    print("Latitude:", lat_array.shape)
    print("Longitude:", lon_array.shape)
    print("Levels:", lev_array.shape)


print_data_structure()


# %% Vertical Distribution Analysis
def plot_vertical_distribution():
    """Compare vertical distribution patterns between particle sizes"""
    small_vars = list(data.small_particles.keys())
    large_vars = list(data.large_particles.keys())

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Quasi-Spherical Particles (0.5-10 μm)",
            "Fiber Particles (35-70 μm)",
        ),
        shared_yaxes=True,
    )

    # Process each particle type
    for i, (particle_type, vars_list) in enumerate(
        [("Small", small_vars), ("Large", large_vars)]
    ):
        for var in vars_list:
            if particle_type == "Small":
                particles = np.array(data.small_particles[var])
            else:
                particles = np.array(data.large_particles[var])

            # Print debug information
            print(f"\nProcessing {particle_type} particles, variable {var}")
            print(f"Shape of particle array: {particles.shape}")

            # Calculate mean profile based on actual data structure
            # First convert to numpy array to ensure we have the data loaded
            particles = np.array(particles)

            # Calculate temporal and spatial means based on the actual dimensions
            if len(particles.shape) == 4:  # If 4D array (time, lev, lat, lon)
                mean_profile = np.mean(
                    np.mean(np.mean(particles, axis=0), axis=1), axis=1
                )
            elif len(particles.shape) == 3:  # If 3D array (time, lat, lon)
                mean_profile = np.mean(np.mean(particles, axis=1), axis=1)
            else:
                print(f"Unexpected shape for {var}: {particles.shape}")
                continue

            fig.add_trace(
                go.Scatter(
                    x=mean_profile,
                    y=lev_array,
                    name=f"{var}",
                    mode="lines",
                ),
                row=1,
                col=i + 1,
            )

    # Update layout with log scale for pressure levels
    fig.update_yaxes(type="log", autorange="reversed", title="Pressure Level (hPa)")
    fig.update_xaxes(title="Mean Concentration")

    fig.update_layout(
        height=600, width=1000, title="Vertical Distribution by Particle Size"
    )

    return fig


# Try plotting with debug information
fig_vertical = plot_vertical_distribution()
fig_vertical.show()


# %% Deposition Analysis
def plot_deposition_comparison():
    """Compare wet and dry deposition rates for different particle sizes"""
    # Get variables for both deposition types
    dry_vars = list(data.dry_deposition.keys())
    wet_vars = list(data.wet_deposition.keys())

    # Print debug information
    print("Dry deposition variables:", dry_vars)
    print("Wet deposition variables:", wet_vars)

    # Create figure with correct number of subplots based on number of variables
    n_vars = len(dry_vars)
    fig = make_subplots(
        rows=n_vars,
        cols=3,
        subplot_titles=["Dry Deposition", "Wet Deposition", "Wet/Dry Ratio"] * n_vars,
        shared_xaxes=True,
        shared_yaxes=True,
    )

    # Process each particle size
    for i, (dry_var, wet_var) in enumerate(zip(dry_vars, wet_vars)):
        # Get deposition data
        dry_dep = np.array(data.dry_deposition[dry_var])
        wet_dep = np.array(data.wet_deposition[wet_var])

        # Calculate time means
        dry_mean = np.mean(dry_dep, axis=0)
        wet_mean = np.mean(wet_dep, axis=0)

        # Calculate zonal means for line plots
        dry_zonal = np.mean(dry_mean, axis=1)
        wet_zonal = np.mean(wet_mean, axis=1)

        # Plot dry deposition
        fig.add_trace(
            go.Scatter(
                x=lat_array,
                y=dry_zonal,
                name=f"{dry_var} - Dry",
            ),
            row=i + 1,
            col=1,
        )

        # Plot wet deposition
        fig.add_trace(
            go.Scatter(
                x=lat_array,
                y=wet_zonal,
                name=f"{wet_var} - Wet",
            ),
            row=i + 1,
            col=2,
        )

        # Calculate and plot wet/dry ratio
        ratio = wet_mean / (dry_mean + 1e-10)  # avoid division by zero
        fig.add_trace(
            go.Heatmap(
                z=ratio,
                x=lon_array,
                y=lat_array,
                colorscale="RdBu",
                colorbar=dict(title="Wet/Dry Ratio"),
                showscale=(i == 0),  # Only show colorbar for first row
            ),
            row=i + 1,
            col=3,
        )

        # Update axes labels
        if i == n_vars - 1:  # Only add labels for bottom row
            fig.update_xaxes(title_text="Latitude", row=i + 1, col=1)
            fig.update_xaxes(title_text="Latitude", row=i + 1, col=2)
            fig.update_xaxes(title_text="Longitude", row=i + 1, col=3)

        fig.update_yaxes(title_text="Deposition Rate", row=i + 1, col=1)

    fig.update_layout(
        height=300 * n_vars,  # Adjust height based on number of variables
        width=1200,
        title="Deposition Patterns by Particle Size",
        showlegend=True,
    )

    return fig


# Add debug print before plotting
print("\nStarting deposition plot...")
fig_deposition = plot_deposition_comparison()
fig_deposition.show()


# %% Transport and Settling Analysis
def plot_transport_settling():
    """Analyze transport patterns and settling behavior"""
    small_vars = list(data.small_particles.keys())
    large_vars = list(data.large_particles.keys())

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            f"Size Bin {i+1}" for i in range(len(small_vars) + len(large_vars))
        ],
        shared_xaxes=True,
        shared_yaxes=True,
    )

    # Process each particle size bin
    for i, var in enumerate(small_vars + large_vars):
        row = (i // 3) + 1
        col = (i % 3) + 1

        if i < len(small_vars):
            particles = np.array(data.small_particles[var])
        else:
            particles = np.array(data.large_particles[var])

        # Calculate temporal mean first
        mean_time = np.mean(particles, axis=0)  # average over time
        # Then calculate zonal mean
        mean_dist = np.mean(mean_time, axis=2)  # average over longitude

        fig.add_trace(
            go.Heatmap(
                z=mean_dist,
                x=lat_array,
                y=lev_array,
                colorscale="Viridis",
                colorbar=dict(title="Concentration"),
            ),
            row=row,
            col=col,
        )

        # Update axes
        fig.update_yaxes(
            type="log", autorange="reversed", title="Pressure (hPa)" if col == 1 else ""
        )
        fig.update_xaxes(title="Latitude" if row == 2 else "")

    fig.update_layout(
        height=800, width=1200, title="Transport and Settling Patterns by Size Bin"
    )

    return fig


fig_transport = plot_transport_settling()
fig_transport.show()


# %% Size-Dependent Characteristics
def analyze_size_characteristics():
    """Analyze key characteristics based on particle size"""
    small_vars = list(data.small_particles.keys())
    large_vars = list(data.large_particles.keys())

    # Calculate mean height for each size bin
    heights = []
    concentrations = []

    for vars_list in [small_vars, large_vars]:
        for var in vars_list:
            if var in small_vars:
                particles = np.array(data.small_particles[var])
            else:
                particles = np.array(data.large_particles[var])

            # Calculate temporal mean first
            mean_time = np.mean(particles, axis=0)
            # Then calculate horizontal mean
            weights = np.mean(mean_time, axis=(1, 2))
            avg_height = np.average(lev_array, weights=weights)
            heights.append(avg_height)

            # Calculate mean concentration
            concentrations.append(np.mean(particles))

    # Create summary plot
    fig = make_subplots(rows=1, cols=2)

    # Plot average heights
    fig.add_trace(
        go.Bar(
            x=["Bin 1", "Bin 2", "Bin 3", "Bin 4", "Bin 5", "Bin 6"],
            y=heights,
            name="Mean Height",
        ),
        row=1,
        col=1,
    )

    # Plot mean concentrations
    fig.add_trace(
        go.Bar(
            x=["Bin 1", "Bin 2", "Bin 3", "Bin 4", "Bin 5", "Bin 6"],
            y=concentrations,
            name="Mean Concentration",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=500, width=1000, title="Size-Dependent Characteristics", showlegend=False
    )

    return fig


fig_characteristics = analyze_size_characteristics()
fig_characteristics.show()
