"""Python implementation of FLINT: Fast Laplace-like INverTer (2D) implementation.

This module provides a Python implementation of FLINT, a fast algorithm for estimating
2D NMR relaxation distributions. The algorithm is based on the work of Paul Teal and
C. Eccles, who developed an adaptive truncation method for matrix decompositions to
efficiently estimate NMR relaxation distributions.

For more information on FLINT, see:
- https://github.com/paultnz/flint
- P.D. Teal and C. Eccles. Adaptive truncation of matrix decompositions and efficient
  estimation of NMR relaxation distributions. Inverse Problems, 31(4):045010, April
  2015. http://dx.doi.org/10.1088/0266-5611/31/4/045010
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import numpy as np
import plotly.graph_objects as go  # type: ignore
import plotly.io as pio  # type: ignore
from plotly.subplots import make_subplots  # type: ignore


def kernel_t2(tau: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """T2 exponential decay."""
    return np.exp(-np.outer(tau, 1 / t2))


def kernel_t1_IR(tau: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """T1 exponential decay for Inversion Recovery experiments."""
    return 1 - 2 * np.exp(-np.outer(tau, 1 / t1))


def kernel_t1_SR(tau: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """T1 exponential decay for Saturation Recovery experiments."""
    return 1 - 1 * np.exp(-np.outer(tau, 1 / t1))


def logarithm_t_range(t_range: np.ndarray, kernel_dim: int) -> np.ndarray:
    """Generates a logarithmic time range."""
    return np.logspace(np.log10(t_range[0]), np.log10(t_range[1]), num=kernel_dim)


class NMRsignal:
    """Represents an NMR signal with time constants and signal amplitudes."""

    def __init__(
        self, signal: np.ndarray, tau1: np.ndarray, tau2: Optional[np.ndarray] = None
    ) -> None:
        """Initialize an NMRsignal object.

        Args:
            tau1 (np.ndarray): 1D array of NMRsignal first time axis.
            tau2 (np.ndarray): 1D array of NMRsignal second time axis.
            signal (np.ndarray): 2D array of signal (complex).

        Raises:
            ValueError: If the signal has more than 2 dimensions.
        """
        self.tau1 = tau1
        self.tau2 = tau2
        if signal.ndim == 1 or signal.ndim == 2:
            self.signal = np.real(signal)
        else:
            raise ValueError("signal must be either a 1D or 2D array")

    @classmethod
    def load_from_data(
        cls, signal: np.ndarray, tau1: np.ndarray, tau2: Optional[np.ndarray] = None
    ) -> NMRsignal:
        """Constructs an NMRsignal object from time constants and signal amplitudes."""
        if signal.ndim == 1:
            signal = signal.reshape(signal.shape[0], 1)
        if tau1.ndim != 1 or (tau2 is not None and tau2.ndim != 1):
            raise ValueError("tau1 and tau2 (if provided) must be 1D arrays.")
        if signal.ndim not in (1, 2):
            raise ValueError("signal must be either a 1D or 2D array")
        if tau2 is None:
            return cls(signal, tau1, None)
        if signal.shape[0] != tau1.size or signal.shape[1] != tau2.size:
            raise ValueError("tau1, tau2, and signal must have compatible dimensions.")
        return cls(signal, tau1, tau2)

    @classmethod
    def load_from_txtfile(cls, file_path: str) -> NMRsignal:
        """Constructs an NMRsignal object from a file."""
        data = np.loadtxt(file_path)
        # index = data[:,0]
        tau1 = data[:, 1]
        signal = data[:, 2] + 1j * data[:, 3] if data.shape[1] > 2 else data[:, 1:]
        tau2 = None
        return cls.load_from_data(signal, tau1, tau2)


class Flint:
    """
    A class for performing 1D/2D Inverse Laplace Transform of NMR data.

    Attributes:
        nmr_signal (NMRsignal): The 2D array of NMR signal to be processed.
        kernel_shape (tuple[int, int]): The dimensions of the 2D kernel,
          given as (t1kernel_dim, t2kernel_dim).
        kernel_name (str): The name of the kernel function to be used for
          the inverse Laplace transform.
            Valid options include: "T1IRT2", "T1SRT2", "T2T2", "T1IR", "T1SR", and "T2".
        alpha (float): The (Tikhonov) regularization parameter.
        t1range (Optional[np.ndarray]): The range of T1 relaxation times,
          given as [t1start, t1finish]. Defaults to None.
        t2range (Optional[np.ndarray]): The range of T2 relaxation times,
          given as [t2start, t2finish]. Defaults to None.
        SS (Optional[np.ndarray]): An optional starting estimate.
          Defaults to an array of ones with shape dimKernel2D.
        tol (float): The relative change between successive calculations for exit.
          Defaults to 1e-5.
        maxiter (int): The maximum number of iterations. Defaults to 100001.
        progress (int): The number of iterations between progress displays.
        Defaults to 500. Should be at least several hundred because calculating
          the error is slow.
    """

    def __init__(
        self,
        nmr_signal: NMRsignal,
        kernel_shape: tuple[int, int],
        kernel_name: str,
        alpha: float,
        t1range: Optional[np.ndarray],
        t2range: Optional[np.ndarray] = None,
        SS: Optional[np.ndarray] = None,
        tol: float = 1e-5,
        maxiter: int = 100001,
        progress: int = 500,
    ) -> None:
        """Initialize a new Flint object.

        Args:
            nmr_signal (NMRsignal): The 2D array of NMR signal to be processed.
            kernel_shape (Tuple[int, int]): The dimensions of the 2D kernel.
            kernel_name (str): The name of the kernel function to be used.
            alpha (float): The (Tikhonov) regularization parameter.
            t1range (Optional[np.ndarray]): The range of T1 relaxation times.
            t2range (Optional[np.ndarray]): The range of T2 relaxation times.
            SS (Optional[np.ndarray]): An optional starting estimate.
            tol (float): The relative change between successive calculations for exit.
            maxiter (int): The maximum number of iterations. Defaults to 100001.
            progress (int): The number of iterations between progress displays.
        """
        kernel_functions: dict[str, list] = {
            "T1IRT2": [kernel_t1_IR, kernel_t2],
            "T1SRT2": [kernel_t1_SR, kernel_t2],
            "T2T2": [kernel_t2, kernel_t2],
            "T1IR": [kernel_t1_IR],
            "T1SR": [kernel_t1_SR],
            "T2": [kernel_t2],
        }

        self.signal = nmr_signal
        self.kernel_type = kernel_name
        self.alpha = alpha
        self.tol = tol
        self.maxiter = maxiter
        self.progress = progress
        self.resida = np.full((maxiter), np.nan)
        self.dim_kernel2d = kernel_shape

        if kernel_name in kernel_functions:
            kernel_function = kernel_functions[kernel_name]

            if len(kernel_function) == 2 and t1range and t2range and self.signal.tau2:
                self.t1axis = logarithm_t_range(t1range, kernel_shape[0])
                self.t2axis = logarithm_t_range(t2range, kernel_shape[1])
                self.t1kernel = self.set_kernel(kernel_function[0], self.signal.tau1, self.t1axis)
                self.t2kernel = self.set_kernel(kernel_function[1], self.signal.tau2, self.t2axis)
            elif len(kernel_function) == 1 and t1range:
                self.t1axis = logarithm_t_range(t1range, kernel_shape[0])
                self.t2axis = np.array([1])
                self.t1kernel = self.set_kernel(kernel_function[0], self.signal.tau1, self.t1axis)
                self.t2kernel = np.identity(1)

    def solve_flint(self, SS: Optional[np.ndarray] = None) -> None:
        """
        Solves the Flint method.

        Args:
            SS (Optional[np.ndarray]): An optional starting estimate.

        """
        if SS is None:
            SS = np.ones((self.dim_kernel2d[0], self.dim_kernel2d[1]))

        self.SS = SS

        t1kernel_operator = self.t1kernel.T @ self.t1kernel
        t2kernel_operator = self.t2kernel.T @ self.t2kernel
        signal_operator = self.t1kernel.T @ self.signal.signal @ self.t2kernel
        signal_trace = np.trace(
            self.signal.signal @ self.signal.signal.T
        )  # used for calculating residual

        lipschitz_constant = self.calculate_lipschitz_constant(
            t1kernel_operator, t2kernel_operator
        )

        YY = self.SS
        tt = 1
        factor1 = (lipschitz_constant - 2 * self.alpha) / lipschitz_constant  # equation factor 1
        factor2 = 2 / lipschitz_constant  # equation factor 2
        lastres = np.inf

        for iteration in range(self.maxiter):
            term2 = signal_operator - t1kernel_operator @ YY @ t2kernel_operator
            Snew = factor1 * YY + factor2 * term2
            Snew[Snew < 0] = 0.0
            ttnew = 0.5 * (1 + np.sqrt(1 + 4 * tt**2))
            trat = (tt - 1) / ttnew
            YY = Snew + trat * (Snew - self.SS)
            tt = ttnew
            self.SS = Snew

            if iteration % self.progress == 0:
                # Don't calculate the residual every iteration; it takes much longer
                # than the rest of the algorithm
                normS = self.alpha * np.sum(self.SS**2)
                resid = (
                    signal_trace
                    - 2 * np.trace(self.SS.T @ signal_operator)
                    + np.trace(self.SS.T @ t1kernel_operator @ self.SS @ t2kernel_operator)
                    + normS
                )
                self.resida[iteration] = resid
                resd = (lastres - resid) / resid
                lastres = resid
                # show progress
                # print("%7d % 1.2e % 1.2e % 1.4e % 1.4e" % (iteration, tt, 1 - trat, resid, resd))
                if np.abs(resd) < self.tol:
                    break

    def plot(self) -> go.Figure:
        """Plots the result of the inverse Laplace transform.

        Returns:
            A plotly figure object.
        """
        plotting_functions = {
            "T2": plot_T2_ILT,
            "T1IR": plot_T2_ILT,
            "T1SR": plot_T2_ILT,
        }

        if self.kernel_type in plotting_functions:
            figure = plotting_functions[self.kernel_type](self.SS, self.t1axis, self.t2axis)
            return figure

    def calculate_lipschitz_constant(self, K1K1: np.ndarray, K2K2: np.ndarray) -> float:
        """Calculates the Lipschitz constant for the given kernel operators `K1K1` and `K2K2`.

        Args:
            K1K1 (np.ndarray): kernel operator 1.
            K2K2 (np.ndarray): kernel operator 2.

        Returns:
            float: Lipschitz constant.
        """
        SS: np.ndarray = np.copy(self.SS)
        LL = np.inf
        MAX_ITERATIONS: int = 100
        for _ii in range(MAX_ITERATIONS):
            lastLL = LL
            LL = np.sqrt(np.sum(SS**2))
            if np.abs(LL - lastLL) / LL < 1e-10:
                break
            SS = self.update_SL(SS, K1K1, K2K2, LL)
        LL = 1.001 * 2 * (LL + self.alpha)
        print("Lipschitz constant found:", LL)
        return LL

    def update_SL(
        self, SL: np.ndarray, K1K1: np.ndarray, K2K2: np.ndarray, LL: float
    ) -> np.ndarray:
        """
        Update the SVD coefficients of the SS matrix.

        Args:
            SL (np.ndarray): The SVD coefficients of the SS matrix.
            K1K1 (np.ndarray): The Gram matrix of the t1 kernel.
            K2K2 (np.ndarray): The Gram matrix of the t2 kernel.
            LL (float): The Lipschitz constant.

        Returns:
            np.ndarray: The updated SVD coefficients of the SS matrix.
        """
        SL = SL / LL
        SL = K1K1 @ SL @ K2K2
        return SL

    def set_kernel(
        self,
        kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
        tau: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        """Sets a kernel for given tau and T arrays."""
        K = kernel(tau, t)
        return K


def plot_T2_ILT(SS: np.ndarray, t1axis: np.ndarray, t2axis: Optional[np.ndarray] = None) -> None:
    """
    Plot the 1D Inverse Laplace Inversion - T2 decay.

    Args:
        SS (np.ndarray): Starting estimate.
        t1axis (np.ndarray): The T1 relaxation axis.
        t2axis (Optional[np.ndarray]): The T2 relaxation axis. Defaults to None.
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
    # Show the figure
    fig.show()


def plot_T1IR_ILT(SS: np.ndarray, t1axis: np.ndarray, t2axis: Optional[np.ndarray] = None) -> None:
    """
    Plot the 1D Inverse Laplace Inversion - T1IR decay.

    Args:
        SS (np.ndarray): Starting estimate.
        t1axis (np.ndarray): The T1 relaxation axis.
        t2axis (Optional[np.ndarray]): The T2 relaxation axis. Defaults to None.
    """
    plot_T2_ILT(SS, t1axis, t2axis=None)


def generate_smilyface_signal(
    N1: int, N2: int, Nx: int, Ny: int, tau1min: float, tau1max: float, deltatau2: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def generate_t2_distribution_signal_decay(
    signal_num_points: int,
    echo_time: float,
    normalized_noise: float,
    t2_distribution_dimension: int,
    t2_distribution_axislim: np.ndarray,
    amplitudes: np.ndarray,
    centers: np.ndarray,
    widths: np.ndarray,
    plot: Optional[bool] = False,
) -> tuple:
    """Generates a 1D NMR signal with noise from a 'true' T2 relaxation time distribution.

    Args:
        signal_num_points (int): number of time points to use in the NMR signal.
        echo_time (float): time between consecutive echo signals in the NMR signal.
        normalized_noise(float): normalized signal noise.
        t2_distribution_dimension(int): dimension of t2 relaxation time distribution.
        t2_distribution_axislim(np.array): time limits of the t2 relaxation time distribution.
        amplitudes (np.array): amplitudes of the Gaussian functions.
        centers (np.array): centers of the Gaussian functions.
        widths (np.array): widths of the Gaussian functions.
        plot (bool, optional): whether to plot the true relaxation map and NMR signal with noise.

    Returns:
        signal_time_axis (np.array): time points used in the NMR signal.
        signal_with_noise (np.array): NMR signal with noise.
        t2_distribution_time_axis (np.array): time points used in the true relaxation map.
        t2_distribution_intensity (np.array): true relaxation map that resembles a smiley face.
    """
    signal_time_axis = (1 + np.arange(signal_num_points)) * echo_time
    ILT_time_axis = np.logspace(
        np.log10(t2_distribution_axislim[0]),
        np.log10(t2_distribution_axislim[1]),
        t2_distribution_dimension,
    )
    kernel = np.exp(-np.outer(signal_time_axis, 1 / ILT_time_axis))

    # building T2 time distribution
    t2_distribution_intensity = np.zeros((t2_distribution_dimension))

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
