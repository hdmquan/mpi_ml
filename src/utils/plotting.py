import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.utils import PATH


def plot_layer(x, lev, num_channels=6, titles=None, save=False):
    """
    Shape: [batch_size, channels, altitude, latitude, longitude]
    """
    rows = (num_channels + 1) // 2
    if titles is None:
        titles = [f"Bin {i+1}" for i in range(num_channels)]

    fig = make_subplots(
        rows=rows,
        cols=2,
        subplot_titles=titles,
    )

    for i in range(num_channels):
        # :D
        row = i // 2 + 1
        col = i % 2 + 1

        fig.add_trace(
            go.Heatmap(
                z=x[0, i, lev, :, :],
                colorscale="RdBu",
                reversescale=True,
                # zmin=-1,
                # zmax=1,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        height=800,
    )

    if save:
        fig.write_html(PATH.SRC / f"layer_{lev}.html")
    # else:
    fig.show()


def plot_long_cut(x, long, num_channels=6, titles=None, save=False):
    """
    Shape: [batch_size, channels, altitude, latitude, longitude]
    """
    rows = (num_channels + 1) // 2
    if titles is None:
        titles = [f"Bin {i+1}" for i in range(num_channels)]

    fig = make_subplots(rows=rows, cols=2, subplot_titles=titles)

    for i in range(num_channels):
        # :D
        row = i // 2 + 1
        col = i % 2 + 1

        fig.add_trace(
            go.Heatmap(
                z=x[0, i, :, :, long],
                colorscale="RdBu",
                reversescale=True,
                # zmin=-1,
                # zmax=1,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        height=800,
    )

    if save:
        fig.write_html(PATH.SRC / f"long_cut_{long}.html")
    # else:
    fig.show()
