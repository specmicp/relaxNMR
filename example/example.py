from relaxNMR.io import read_folder
from relaxNMR.core.ilt import ILT_fit, ILT_fit_collection

import matplotlib.pyplot as plt

# preprocessing
signals = read_folder("OPC/", ".dps")
signals.phase(1,3)
signals.remove_first_echo()
average_signal = signals.average()

Trange=(1e-5, 1) # the range of relaxation times
alpha = 1e-4     # the penalization parameter

# fit individual experiment
fitted = ILT_fit_collection(signals, Trange, alpha)
fitted.normalize()

# fit average
fitted_average = ILT_fit(average_signal, Trange, alpha)
fitted_average.normalize()


# plot
fig, ax = plt.subplots(constrained_layout = True)

for s in signals:
    ax.semilogx(s.tau, s.real, "-", lw=1, color="#666666")

ax.semilogx(average_signal.tau, average_signal.real, "-", lw=2, color="#dd4444")
ax.set_ylabel("Signal intensity")
ax.set_xlabel(r"$\tau$ (s) ")

fig, ax = plt.subplots(constrained_layout = True)

for f in fitted:
    ax.semilogx(f.Ts, f.magnitude, "-", lw=1, color="#666666")

ax.semilogx(fitted_average.Ts, fitted_average.magnitude, "-", lw=2, color="#dd4444")
ax.set_ylabel("Normalized magnitude")
ax.set_xlabel(r"Relaxation times (s)")