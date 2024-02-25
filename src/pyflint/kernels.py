"""Kernel functions for Flint class."""

from typing import Callable

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


def set_kernel(
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    tau: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Sets a kernel for given tau and T arrays."""
    K = kernel(tau, t)
    return K
