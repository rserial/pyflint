"""Python implementation o FLINT: Fast Laplace-like INverTer (2D) implementation.

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

    def __init__(self, tau1: np.ndarray, tau2: np.ndarray, signal: np.ndarray) -> None:
        """Initialize an NMRsignal object.

        Args:
            tau1 (np.ndarray): 1D array of NMRsignal first time axis.
            tau2 (np.ndarray): 1D array of NMRsignal second time axis.
            signal (np.ndarray): 2D array of signal (complex).
        """
        self.tau1 = tau1
        self.tau2 = tau2
        self.signal = np.real(signal)

    @classmethod
    def from_data(cls, tau1: np.ndarray, tau2: np.ndarray, signal: np.ndarray) -> NMRsignal:
        """Constructs an NMRsignal object from time constants and signal amplitudes."""
        if tau1.ndim != 1 or tau2.ndim != 1 or signal.ndim != 2:
            raise ValueError("tau1, tau2, and signal must be 1D, 1D, and 2D arrays, respectively.")
        if tau1.size != signal.shape[0] or tau2.size != signal.shape[1]:
            raise ValueError("tau1, tau2, and signal must have compatible dimensions.")
        return cls(tau1, tau2, signal)


class Flint:
    """Use this class to perform 1D/2D Inverse Laplace Transform of NMR data."""

    def __init__(
        self,
        K1range: np.ndarray,
        K2range: np.ndarray,
        dimKernel2D: np.ndarray,
        nmr_signal: NMRsignal,
        alpha: float,
        SS: Optional[np.ndarray] = None,
        tol: float = 1e-5,
        maxiter: int = 100001,
        progres: int = 500,
    ) -> None:
        """
        FLINT: Fast Lapace-like INverTer (2D).

        Args:
            K1range (np.array): The T1 relaxation kernel matrix (size Nechos x NT1)
                (set this to 1 if processing a 1D T2 experiment)
            K2range (np.array): The T2 relaxation kernel matrix (size Nechodelays x NT2)
                (set this to 1 if processing a 1D T1 experiment)
            dimKernel2D (np.array): ...
            nmr_signal (NMRsignal): ...
            alpha (float): ...
            SS (Optional[np.ndarray]): ...
            tol (float): ...
            maxiter (int): ...
            progres (int): ...
        """
        self.K1axis = logarithm_t_range(K1range, dimKernel2D[0])
        self.K2axis = logarithm_t_range(K2range, dimKernel2D[1])
        self.signal = nmr_signal

        self.alpha = alpha
        self.K1 = self.set_kernel(kernel_t1_IR, self.signal.tau1, self.K1axis)
        self.K2 = self.set_kernel(kernel_t2, self.signal.tau2, self.K2axis)

        if SS is None:
            SS = np.asarray([])
        self.SS = SS if SS.size != 0 else np.ones((self.K1.shape[1], self.K2.shape[1]))
        self.tol = tol
        self.maxiter = maxiter
        self.progres = progres
        self.resida = np.full((maxiter), np.nan)

    def set_kernel(
        self,
        kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
        tau: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        """Sets a kernel for given tau and T arrays."""
        K = kernel(tau, t)
        return K
