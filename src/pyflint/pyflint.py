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

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go  # type: ignore

import pyflint.plot as plot
from pyflint import kernels
from pyflint.utils import logarithm_t_range


def perform_ilt_and_plot(
    decay: np.ndarray,
    tau1: np.ndarray,
    dimKernel2D: Tuple[int, int],
    alpha: float,
    kernel_name: str,
    t1_range: Tuple[float, float],
    t2_range: Optional[Tuple[float, float]],
    plot_title: str,
    tau2: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform inverse Laplace transform (ILT) from given data.

    Args:
        decay (np.ndarray): Decay data.
        tau1 (np.ndarray): Value of tau1 parameter.
        dimKernel2D (int): Dimension of the 2D kernel.
        alpha (float): Alpha parameter for ILT.
        kernel_name (str): Name of the kernel.
        t1_range (Tuple[float, float]): Range of T1 values.
        t2_range (Tuple[float, float]): Range of T2 values.
        plot_title (str): Title for the plot.
        tau2 (Optional[np.ndarray]): Value of tau2 parameter.

    Returns:
        Tuple[np.ndarray, np.ndarray]: ilt_t1_axis and corresponding ilt_data.
    """
    signal = FlintSignal.load_from_data(decay, tau1, tau2)

    flint = Flint(signal, dimKernel2D, kernel_name, alpha, t1_range, t2_range)
    flint.solve_flint()
    fig_samplebase = flint.plot()
    fig_samplebase.update_layout(title_text=plot_title)
    fig_samplebase.show()

    ilt_t1_axis = np.squeeze(flint.t1axis)
    ilt_data = np.squeeze(flint.SS)
    return ilt_t1_axis, ilt_data


class FlintSignal:
    """Represents a Flint compatible signal with time constants and signal amplitudes."""

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
    ) -> FlintSignal:
        """Constructs an FlintSignal object from time constants and signal amplitudes."""
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
    def load_from_1d_txtfile(cls, file_path: str) -> FlintSignal:
        """Constructs an FlintSignal object from a file."""
        data = np.loadtxt(file_path)
        tau1 = data[:, 0]
        signal = data[:, 1] + 1j * data[:, 2] if data.shape[1] > 2 else data[:, 1] + 1j * 0
        tau2 = None
        return cls.load_from_data(signal, tau1, tau2)

    def save_to_1d_txtfile(self, file_path: str) -> None:
        """Saves the NMRsignal object to a 1D txt file."""
        signal = self.signal.ravel()
        tau1 = self.tau1
        data = np.column_stack((tau1, signal))
        np.savetxt(file_path, data)

    @classmethod
    def average_signals(cls, dir_list: List[str]) -> FlintSignal:
        """Averages a list of NMR signals from 1D txt files.

        Args:
            dir_list (List[str]): A list of file paths.

        Returns:
            FlintSignal: An FlintSignal object with the averaged signal.
        """
        signals = [cls.load_from_1d_txtfile(dir_path) for dir_path in dir_list]

        average_real = np.zeros(signals[0].signal.size)
        average_imag = np.zeros(signals[0].signal.size)

        for signal in signals:
            average_real += signal.signal.real.reshape(average_real.shape)
            average_imag += signal.signal.imag.reshape(average_imag.shape)
        average_signal = average_real + 1j * average_imag

        average_signal /= len(signals)
        tau1 = signals[0].tau1
        tau2 = signals[0].tau2
        return cls.load_from_data(average_signal, tau1, tau2)


class Flint:
    """
    A class for performing 1D/2D Inverse Laplace Transform of NMR data.

    Attributes:
        flint_signal (NMRsignal): The 2D array of NMR signal to be processed.
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
        flint_signal: FlintSignal,
        kernel_shape: tuple[int, int],
        kernel_name: str,
        alpha: float,
        t1range: Optional[Tuple[float, float]],
        t2range: Optional[Tuple[float, float]] = None,
        SS: Optional[np.ndarray] = None,
        tol: float = 1e-5,
        maxiter: int = 100001,
        progress: int = 500,
    ) -> None:
        """Initialize a new Flint object.

        Args:
            flint_signal (FlintSignal): The 2D array of NMR signal to be processed.
            kernel_shape (Tuple[int, int]): The dimensions of the 2D kernel.
            kernel_name (str): The name of the kernel function to be used.
            alpha (float): The (Tikhonov) regularization parameter.
            t1range (Optional[Tuple[float, float]]): The range of T1 relaxation times.
            t2range (Optional[Tuple[float, float]]): The range of T2 relaxation times.
            SS (Optional[np.ndarray]): An optional starting estimate.
            tol (float): The relative change between successive calculations for exit.
            maxiter (int): The maximum number of iterations. Defaults to 100001.
            progress (int): The number of iterations between progress displays.

        Raises:
            ValueError: If kernel_name is not in kernel_functions dictionary.
        """
        kernel_functions: dict[str, list] = {
            "T1IRT2": [kernels.kernel_t1_IR, kernels.kernel_t2],
            "T1SRT2": [kernels.kernel_t1_SR, kernels.kernel_t2],
            "T2T2": [kernels.kernel_t2, kernels.kernel_t2],
            "T1IR": [kernels.kernel_t1_IR],
            "T1SR": [kernels.kernel_t1_SR],
            "T2": [kernels.kernel_t2],
        }

        self.signal = flint_signal
        self.kernel_type = kernel_name
        self.alpha = alpha
        self.tol = tol
        self.maxiter = maxiter
        self.progress = progress
        self.resida = np.full((maxiter), np.nan)
        self.dim_kernel2d = kernel_shape

        if kernel_name not in kernel_functions:
            available_options = ", ".join(kernel_functions.keys())
            raise ValueError(
                f"Invalid kernel name '{kernel_name}'. Available options are: {available_options}"
            )

        if kernel_name in kernel_functions:
            kernel_function = kernel_functions[kernel_name]

            if (
                len(kernel_function) == 2
                and t1range is not None
                and t2range is not None
                and self.signal.tau2 is not None
            ):
                self.t1axis = logarithm_t_range(t1range, kernel_shape[0])
                self.t2axis = logarithm_t_range(t2range, kernel_shape[1])
                self.t1kernel = self.set_kernel(kernel_function[0], self.signal.tau1, self.t1axis)
                self.t2kernel = self.set_kernel(kernel_function[1], self.signal.tau2, self.t2axis)
            elif len(kernel_function) == 1 and t1range is not None:
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

    def plot(self) -> go.Figure:
        """Plots the result of the inverse Laplace transform.

        Returns:
            A plotly figure object.
        """
        plotting_functions: Dict[str, Callable[..., Tuple[Any, ...]]] = {
            "T2": plot.plot_T2_ILT,
            "T1IR": plot.plot_T1IR_ILT,
            "T1SR": plot.plot_T1SR_ILT,
            "T1IRT2": plot.plot_2DILT,
        }

        if self.kernel_type in plotting_functions:
            figure = plotting_functions[self.kernel_type](self.SS, self.t1axis, self.t2axis)
            return figure

    def save_to_par_file(self, file_path: str) -> None:
        """Saves FLINT parameters and array data to a parameter (par) file.

        Args:
            file_path (str): The path to save the par file.
        """
        with open(file_path, "w") as par_file:
            # Write FLINT parameters
            par_file.write("# FLINT Parameters\n")
            par_file.write(f"# Date: {datetime.now().strftime('%Y-%m-%d')}\n")
            par_file.write(f"# Kernel name: {self.kernel_type}")
            par_file.write(f"# Dimension: {self.dim_kernel2d[0]}x{self.dim_kernel2d[1]}\n")
            par_file.write(f"# Alpha: {self.alpha}\n")
            par_file.write(f"# Residuals: {self.resida}\n")

            # Write T1 and T2 axis if available
            if self.t1axis:
                par_file.write(f"# t1 axis: {self.t1axis}\n")
            if self.t2axis:
                par_file.write(f"# t2 axis: {self.t2axis}\n")

            # Write array data
            par_file.write("array_data:\n")
            np.savetxt(par_file, self.SS, delimiter=",", fmt="%f")

    def load_from_par_file(
        self, file_path: str
    ) -> Tuple[dict, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Loads parameters, array data, t1axis, and t2axis from a parameter (par) file.

        Args:
            file_path (str): The path to the par file.

        Returns:
            Tuple[dict, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
                A tuple containing a dictionary of parameters, the array data, t1axis, and t2axis.
        """
        ilt_parameters = {}
        ilt_array_data = None
        ilt_t1axis = None
        ilt_t2axis = None

        with open(file_path, "r") as par_file:
            for line in par_file:
                line = line.strip()
                if not line or line.startswith("#") or line.lower() == "arraydata:":
                    continue

                if line.lower().startswith("array_data:"):
                    # Array data section begins, read array data
                    ilt_array_data = np.loadtxt(par_file, delimiter=",")
                    break

                # Check for t1axis and t2axis
                if line.lower().startswith("# t1 axis:"):
                    ilt_t1axis = np.array(line.split(":")[1].strip().split(","), dtype=float)
                    continue
                elif line.lower().startswith("# t2 axis:"):
                    ilt_t2axis = np.array(line.split(":")[1].strip().split(","), dtype=float)
                    continue

                # Split the line into key-value pairs
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()

                # Store the parameters in a dictionary
                ilt_parameters[key] = value

        return ilt_parameters, ilt_array_data, ilt_t1axis, ilt_t2axis

    def set_kernel(
        self,
        kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
        tau: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        """Sets a kernel for given tau and T arrays."""
        K = kernel(tau, t)
        return K
