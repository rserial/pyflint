"""Plotting functions for pyflint."""

from typing import Optional

import numpy as np
import plotly.graph_objects as go  # type: ignore
import plotly.io as pio  # type: ignore
from plotly.subplots import make_subplots  # type: ignore


def plot_T2_ILT(
    SS: np.ndarray, t1axis: np.ndarray, t2axis: Optional[np.ndarray] = None
) -> go.Figure:
    """
    Plot the 1D Inverse Laplace Inversion - T2 decay.

    Args:
        SS (np.ndarray): Starting estimate.
        t1axis (np.ndarray): The T1 relaxation axis.
        t2axis (Optional[np.ndarray]): The T2 relaxation axis. Defaults to None.

    Returns:
        fig(go.Figure): output figure
    """
    # Define the data
    x = t1axis.squeeze()
    y = SS.squeeze()

    # Create the trace
    trace = go.Scatter(x=x, y=y, mode="lines")

    # Create the layout
    layout = go.Layout(
        title="1D Inverse Laplace Inversion - T2 decay",
        xaxis=dict(
            title="Relaxation times (s)",
            type="log",
            tickformat=".1e",
            tickvals=[
                round(x, 4) for x in np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), 4)
            ],
        ),
        yaxis=dict(title="Normalized magnitude"),
        width=500,
        height=500,
    )

    # Create the figure object
    fig = go.Figure(data=[trace], layout=layout)

    fig.update_layout(width=500, height=500, template="pyflint_plotly_template")
    return fig


def plot_2DILT(
    SS: np.ndarray, t1axis: np.ndarray, t2axis: Optional[np.ndarray] = None, ncontours: int = 10
) -> go.Figure:
    """
    Plot a 2D ILT map.

    Args:
        SS (np.ndarray): 2D array of intensity values.
        t1axis (np.ndarray): Array representing the T1 axis.
        t2axis (Optional[np.ndarray]): Array representing the T2 axis.
        ncontours (int): Number of contours for the plot.

    Returns:
        go.Figure: Plotly figure object.
    """
    fig = make_subplots(rows=1, cols=1)

    # Create an intensity map subplot
    fig.add_trace(
        go.Contour(
            x=t2axis,
            y=t1axis,
            z=SS,
            ncontours=ncontours,
            contours_coloring="lines",
            line_width=2,
            colorscale="Blues",
            reversescale=True,
            colorbar=dict(title="Density"),
        )
    )

    # Set log scale on both axes
    if t2axis is not None:
        fig.update_xaxes(
            type="log",
            title="T2 axis",
            tickmode="linear",
            tickvals=np.logspace(np.log10(t2axis[0]), np.log10(t2axis[-1]), 3),
            tickformat=".1e",
            showline=True,
            linewidth=1,
            linecolor="black",
        )

    if t1axis is not None:
        fig.update_yaxes(
            type="log",
            title="T1 axis",
            tickmode="linear",
            tickvals=np.logspace(np.log10(t1axis[0]), np.log10(t1axis[-1]), 3),
            tickformat=".1e",
            showline=True,
            linewidth=1,
            linecolor="black",
        )

    # Add a line for T1=T2 as a dotted line
    if t2axis is not None and t1axis is not None:
        fig.add_shape(
            type="line",
            x0=t2axis[0],
            y0=t1axis[0],
            x1=t2axis[-1],
            y1=t1axis[-1],
            line=dict(dash="dot", width=1, color="black"),
        )

    fig.update_layout(
        title="Density Intensity Maps",
        width=600,
        height=600,
        xaxis=dict(
            type="log",
            title="T2 axis",
            tickmode="linear",
            tickvals=np.logspace(np.log10(t2axis[0]), np.log10(t2axis[-1]), 3),
            tickformat=".1e",
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
        )
        if t2axis is not None
        else None,
        yaxis=dict(
            type="log",
            title="T1 axis",
            tickmode="linear",
            tickvals=np.logspace(np.log10(t1axis[0]), np.log10(t1axis[-1]), 3),
            tickformat=".1e",
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
        )
        if t1axis is not None
        else None,
        xaxis2=dict(zeroline=False, domain=[0.85, 1], showgrid=False),
    )

    return fig


def plot_T1IR_ILT(
    SS: np.ndarray, t1axis: np.ndarray, t2axis: Optional[np.ndarray] = None
) -> go.Figure:
    """
    Plot the 1D Inverse Laplace Inversion - T1IR decay.

    Args:
        SS (np.ndarray): Starting estimate.
        t1axis (np.ndarray): The T1 relaxation axis.
        t2axis (Optional[np.ndarray]): The T2 relaxation axis. Defaults to None.

    Returns:
        fig(go.Figure): output figure
    """
    fig = plot_T2_ILT(SS, t1axis, t2axis=None)
    fig.update_layout(title="1D Inverse Laplace Inversion - T1 decay")
    return fig


def plot_T1SR_ILT(
    SS: np.ndarray, t1axis: np.ndarray, t2axis: Optional[np.ndarray] = None
) -> go.Figure:
    """
    Plot the 1D Inverse Laplace Inversion - T1IR decay.

    Args:
        SS (np.ndarray): Starting estimate.
        t1axis (np.ndarray): The T1 relaxation axis.
        t2axis (Optional[np.ndarray]): The T2 relaxation axis. Defaults to None.

    Returns:
        fig(go.Figure): output figure
    """
    fig = plot_T1IR_ILT(SS, t1axis, t2axis=None)
    return fig


# Load the simple_white template
template = pio.templates["simple_white"]

# Set the font size for the title and axis labels
template.layout.title.font.size = 20
template.layout.xaxis.title.font.size = 20
template.layout.yaxis.title.font.size = 20
template.layout.xaxis.tickfont.size = 16
template.layout.yaxis.tickfont.size = 16
template.layout.xaxis.linewidth = 1.5
template.layout.yaxis.linewidth = 1.5
template.layout.xaxis.tickwidth = 1.5
template.layout.yaxis.tickwidth = 1.5
template.layout.width = 400
template.layout.height = 400

# template.layout.linewidth = 1.8
# Update the template name to 'my_custom_template'
pio.templates["pyflint_plotly_template"] = template

# Load the updated template
pyflint_plotly_template = pio.templates["pyflint_plotly_template"]

# Set the updated template as the default
pio.templates.default = "pyflint_plotly_template"
