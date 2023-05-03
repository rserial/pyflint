"""This script demonstrates the use of Flint."""

import matplotlib.pyplot as plt
import numpy as np

from pyflint import Flint, NMRsignal, generate_smilyface_signal

# dimensions of NMRsignal
N1 = 100
N2 = 25000
# tau1 limits
tau1min = 1e-3
tau1max = 20
# tau2 spacing
deltatau2 = 5e-4

# dimensions true relaxation map
Nx = 500
Ny = 501

# generate signal
tau1, tau2, signal_with_noise, T1a, T2a, Ftrue = generate_smilyface_signal(
    N1, N2, Nx, Ny, tau1min, tau1max, deltatau2
)

# convert to NMRsignal format
nmr_signal = NMRsignal.from_data(tau1, tau2, signal_with_noise)

# Create a Flint object with the desired parameters
t1range = np.array([1e-1, 3e0])
t2range = np.array([1e-1, 3e0])
dimKernel2D = [100, 100]
alpha = 1e-5
flint = Flint(t1range, t2range, dimKernel2D, nmr_signal, alpha)

# Call the solve method to obtain the inverse Laplace transform
result, t1axis, t2axis, residuals = flint.solve_flint()

# Create a figure with 1 row and 3 columns
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# Plot NMR signal in the first subplot
im = axs[0].pcolormesh(tau2, tau1, np.real(nmr_signal.signal))
axs[0].set_title("Input NMR Signal")
axs[0].set_xlabel("tau 2 (s)")
axs[0].set_ylabel("tau1 (s)")
fig.colorbar(im, ax=axs[0])

# Plot Ftrue in the second subplot
im = axs[1].pcolormesh(T2a, T1a, np.real(Ftrue))
axs[1].set_title("True relaxation map")
axs[1].set_xlabel("Time T2 (s)")
axs[1].set_ylabel("Time T1 (s)")
fig.colorbar(im, ax=axs[1])

# Plot reconstructed signal in the third subplot
im = axs[2].pcolormesh(t2axis, t1axis, np.real(result))
axs[2].set_title("ILT reconstructed relaxation map")
axs[2].set_xlabel("Time T2 (s)")
axs[2].set_ylabel("Time T1 (s)")
fig.colorbar(im, ax=axs[2])

plt.tight_layout()
plt.savefig("result.png")
