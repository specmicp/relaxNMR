import numpy as np
import matplotlib.pyplot as plt

from relaxNMR.core import ComplexSignal
from relaxNMR.core.multi_exponential import fit_multi_exponential_fixT

from numpy.random import default_rng
rng = default_rng()

def fn_mode(t, T2, magnitude):
    return magnitude*np.exp(-t/T2)

def get_decay_fn(modes):
    def func(t):
        return np.sum([fn_mode(t, T2, mag) for T2, mag in modes.values()], axis=0)
    return func


def T2s(modes):
    return [T2 for T2, _ in modes.values()]

def add_error(signal, level):
    return signal + rng.normal(0, level, signal.shape[0])

def dephase(real, imag, theta):
    real1 =  np.cos(theta)*real + np.sin(theta)*imag
    imag1 = -np.sin(theta)*real + np.cos(theta)*imag
    return real1, imag1


if __name__ == "__main__":


    modes = {
        "Interlayer": (1e-4, 0.30),
        "Gel pores":(0.5e-3, 0.35),
        "Interhydrates": (1e-3, 0.30),
        "Capillary": (1e-2, 0.05)
        }

    decay = get_decay_fn(modes)

    fig, ax = plt.subplots()
    taus = np.logspace(-4, -1, num=100)

    ax.semilogx(taus, decay(taus))
    #ax.semilogx(ts, decay(ts), ".")
    ax.set_ylim([0, 1])

    taus = np.logspace(-4, -1, num=40)
    real = decay(taus)
    imag = np.zeros_like(real)
    ax.semilogx(taus, real, ".")

    # signal = ComplexSignal(taus, real, imag)
    # ret, _ = fit_multi_exponential_fixT(signal, 4, T2s(modes))
    # print(ret)

    # real = add_error(real, 0.05)
    # ax.semilogx(taus, real, ".")


    # signal = ComplexSignal(taus, real, imag)
    # ret, _ = fit_multi_exponential_fixT(signal, 4, T2s(modes))
    # print(ret)

    # real1, imag1 = dephase(real, imag, 5*np.pi/180)


    # signal = ComplexSignal(taus, real1, imag1)
    # signal = signal.magnitude()
    # ret, _ = fit_multi_exponential_fixT(signal, 4, T2s(modes))
    # print(ret)

    mode0 = []
    mode1 = []
    mode2 = []
    mode3 = []

    errors = [0.001, 0.005, 0.01, 0.05, 0.1]
    for ind, error in enumerate(errors):
        reale = add_error(real, error)

        real1, imag1 = dephase(reale, imag, 10*np.pi/180)
        signal = ComplexSignal(taus, real1, imag1)
        signal = signal.magnitude()

        line,  = ax.semilogx(taus, 0.1*ind+signal.signal, ".")
        ax.semilogx(taus, 0.1*ind+decay(taus), color=line.get_color())

        ret, _ = fit_multi_exponential_fixT(signal, 4, T2s(modes))
        mode0.append(ret[0])
        mode1.append(ret[1])
        mode2.append(ret[2])
        mode3.append(ret[3])



    ax.set_xlabel("Error scale")
    ax.set_ylabel("Taus")

    fig, ax = plt.subplots()
    ax.semilogx(errors, mode0, color="black")
    ax.semilogx(errors, mode1, color="red")
    ax.semilogx(errors, mode2, color="blue")
    ax.semilogx(errors, mode3, color="green")

    ax.set_xlabel("Error scale")
    ax.set_ylabel("Mode fraction")
