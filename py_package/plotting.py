"""Functionality for plotting."""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np

JHTDB_STATS_FPATH = "data/jhtdb-transitional-bl/all-stats.h5"


def plot_profiles_multiple_x_locations():
    """Plot profiles at multiple x-locations."""
    pass


def plot_heatmap(
    quantity: np.array,
    data: dict,
    ix_min: int = 0,
    iy_max: int = -1,
    diverging: bool = False,
) -> go.Figure:
    """Plot a heatmap of a 2-D array of data matching the shape of the
    JHTDB stats.
    """
    return px.imshow(
        quantity[:iy_max, ix_min:],
        x=data["x"][ix_min:],
        y=data["y"][:iy_max],
        origin="lower",
        aspect="auto",
        color_continuous_midpoint=0.0 if diverging else None,
        color_continuous_scale="RdBu_r" if diverging else None,
    )
