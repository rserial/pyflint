"""Functions for generating syntetic data for pyflint."""

from typing import Optional, Tuple

import numpy as np
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore

from pyflint import kernels


def smilyface_signal(
    N1: int, N2: int, Nx: int, Ny: int, tau1min: float, tau1max: float, deltatau2: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates something.

    Args:
        N1 (int): number of tau1 values to use in the NMR signal.
        N2 (int): number of tau2 values to use in the NMR signal.
        Nx (int): number of T1 values to use in the true relaxation map.
        Ny (int): number of T2 values to use in the true relaxation map.
        tau1min (float): minimum value of tau1 to use in the NMR signal, in seconds.
        tau1max (float): maximum value of tau1 to use in the NMR signal, in seconds.
        deltatau2 (float): spacing between consecutive tau2 values to use in the NMR signal.

    Returns:
        tau1 (1D array of floats): values of tau1 used in the NMR signal.
        tau2 (1D array of floats): values of tau2 used in the NMR signal.
        signal_with_noise (2D array of floats): NMR signal with noise.
        T1a (2D array of floats): matrix of T1 values used in the true relaxation map.
        T2a (2D array of floats): matrix of T2 values used in the true relaxation map.
        Ftrue (2D array of floats): true relaxation map that resembles a smiley face.
    """
    tau1 = np.logspace(np.log10(tau1min), np.log10(tau1max), N1)
    tau2 = (1 + np.arange(N2)) * deltatau2
    T1 = np.logspace(np.log10(1e-3), np.log10(3), Nx)
    T2 = np.logspace(np.log10(1e-3), np.log10(3), Ny)
    K2 = np.exp(-np.outer(tau2, 1 / T2))
    K1 = 1 - 2 * np.exp(-np.outer(tau1, 1 / T1))

    T2a, T1a = np.meshgrid((T2), (T1))

    Ftrue = np.zeros((Nx, Ny))

    shift = 1.9

    centre1 = [-0.5 + shift, -1 + shift]
    radius1 = [0.35, 0.75]
    centre2 = [-1.1 + shift, 0.1 + shift]
    radius2 = 0.2
    centre3 = [0.1 + shift, 0.1 + shift]
    radius3 = 0.2

    dist1 = np.sqrt((T2a - centre1[0]) ** 2 + (T1a - centre1[1]) ** 2)
    dist2 = np.sqrt((T2a - centre2[0]) ** 2 + (T1a - centre2[1]) ** 2)
    dist3 = np.sqrt((T2a - centre3[0]) ** 2 + (T1a - centre3[1]) ** 2)
    ang1 = np.arctan2(T1a - centre1[1], T2a - centre1[0])

    Ftrue[np.logical_and(np.logical_and(radius1[0] < dist1, dist1 < radius1[1]), ang1 <= 0)] = 0.4
    Ftrue[dist2 < radius2] = 0.18
    Ftrue[dist3 < radius3] = 0.58

    signal = K1 @ Ftrue @ K2.T
    sigma = np.max(signal) * 1e-4
    noise = sigma * np.random.randn(signal.shape[0], signal.shape[1])
    signal_with_noise = signal + noise

    return tau1, tau2, signal_with_noise, T1a, T2a, Ftrue


def time_distribution_signal_decay(
    signal_num_points: int,
    signal_time_lim: np.ndarray,
    kernel_name: str,
    normalized_noise: float,
    t_dimension: int,
    t_axis_lim: np.ndarray,
    amplitudes: np.ndarray,
    centers: np.ndarray,
    widths: np.ndarray,
    plot: Optional[bool] = False,
) -> tuple:
    """Generates a 1D NMR signal with noise from a 'true' T2 relaxation time distribution.

    Args:
        signal_num_points (int): number of time points to use in the NMR signal.
        signal_time_lim (np.array): signal time limits. [tinit, tend].
        kernel_name (str): name of the kernel function used.
        normalized_noise(float): normalized signal noise.
        t_dimension(int): dimension of t2 relaxation time distribution.
        t_axis_lim(np.array): time limits of the t2 relaxation time distribution.
        amplitudes (np.array): amplitudes of the Gaussian functions.
        centers (np.array): centers of the Gaussian functions.
        widths (np.array): widths of the Gaussian functions.
        plot (bool, optional): whether to plot the true relaxation map and NMR signal with noise.

    Returns:
        signal_time_axis (np.array): time points used in the NMR signal.
        signal_with_noise (np.array): NMR signal with noise.
        t2_distribution_time_axis (np.array): time points used in the true relaxation map.
        t2_distribution_intensity (np.array): true relaxation map that resembles a smiley face.

    Raises:
        ValueError: If kernel_name is not in kernel_functions dictionary.
    """
    kernel_functions: dict[str, list] = {
        "T1IR": [kernels.kernel_t1_IR],
        "T1SR": [kernels.kernel_t1_SR],
        "T2": [kernels.kernel_t2],
    }

    if kernel_name not in kernel_functions:
        available_options = ", ".join(kernel_functions.keys())
        raise ValueError(
            f"Invalid kernel name '{kernel_name}'. Available options are: {available_options}"
        )

    if kernel_name in kernel_functions:
        kernel_function = kernel_functions[kernel_name]

        if kernel_name == "T2":
            delta_time = (signal_time_lim[1] - signal_time_lim[0]) / signal_num_points
            signal_time_axis = (1 + np.arange(signal_num_points)) * delta_time
        if kernel_name == "T1IR" or "T1SR":
            signal_time_axis = np.logspace(
                np.log10(signal_time_lim[0]),
                np.log10(signal_time_lim[1]),
                signal_num_points,
            )
        ILT_time_axis = np.logspace(
            np.log10(t_axis_lim[0]),
            np.log10(t_axis_lim[1]),
            t_dimension,
        )
        kernel = kernels.set_kernel(kernel_function[0], signal_time_axis, ILT_time_axis)

        # building T2 time distribution
        t2_distribution_intensity = np.zeros((t_dimension))

        # Infer the number of populations from the length of distribution_params
        num_populations = len(amplitudes)
        # Generate the T2 distribution
        for i in range(num_populations):
            t2_distribution_intensity += amplitudes[i] * np.exp(
                -((ILT_time_axis - centers[i]) ** 2) / (2 * widths[i] ** 2)
            )

        signal = t2_distribution_intensity @ kernel.T
        sigma = np.max(signal) * normalized_noise
        noise = sigma * np.random.randn(signal.shape[0])
        signal_with_noise = signal + noise

        if plot:
            # Create a subplot with two plots
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=(
                    "Simulated relaxation time distribution",
                    "Simulated NMR signal + noise",
                ),
                column_widths=[0.5, 0.5],
            )

            # Add the first plot (true relaxation map)
            fig.add_trace(
                go.Scatter(x=ILT_time_axis, y=t2_distribution_intensity, name="T2 distribution"),
                row=1,
                col=1,
            )
            fig.update_xaxes(
                title_text="Time (s)",
                type="log",
                tickvals=[
                    round(x, 4)
                    for x in np.logspace(
                        np.log10(np.min(ILT_time_axis)), np.log10(np.max(ILT_time_axis)), 3
                    )
                ],
                tickformat=".1e",
                row=1,
                col=1,
            )
            fig.update_yaxes(title_text="Intensity (a.u)", row=1, col=1)

            # Add the second plot (NMR signal with noise)
            fig.add_trace(
                go.Scatter(x=signal_time_axis, y=signal_with_noise, name="NMR signal + noise"),
                row=1,
                col=2,
            )
            fig.update_xaxes(title_text="Time (s)", row=1, col=2)
            fig.update_yaxes(title_text="Signal amplitude", row=1, col=2)

            fig.update_layout(width=1000, height=500, template="pyflint_plotly_template")
            fig.show()

    return signal_time_axis, signal_with_noise, ILT_time_axis, t2_distribution_intensity
