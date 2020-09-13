import os.path
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import nnls, curve_fit

def average_magnitude_in_folder(folder):
    all_files = glob(os.path.join(folder,"*.dps"))
    # assume tau is the same
    for file in all_files:
        tau, real, imag = get_data_dps(file)
        signal = magnitude(real, imag)

    return tau, signal

def phase(real, imag, first_echo, nb_echo):

    thetas = []
    for i in np.arange(first_echo, first_echo+nb_echo+1):
        if real[i] < 0:
            thetas.append(np.arctan(imag[i]/real[i]) + np.pi)
        else:
            thetas.append(np.arctan(imag[i]/real[i]))

    theta = np.array(thetas).mean()

    real1 =  np.cos(theta)*real + np.sin(theta)*imag
    imag1 = -np.sin(theta)*real + np.cos(theta)*imag

    return real1, imag1


def magnitude(real, imag):
    return np.sqrt(np.power(real, 2)+np.power(imag, 2))



def kernel_T2(tau, T):
    return np.exp(-tau/T)

def prepare_nnls(tau, signal, min_T, max_T, nb_T, alpha):
    Ts = np.logspace(np.log10(min_T), np.log10(max_T), num=nb_T)
    nb_tau = tau.shape[0]
    K = np.zeros((nb_tau+nb_T,nb_T))
    for j in range(nb_T):
        K[:nb_tau,j] = kernel_T2(tau, Ts[j])

    K[nb_tau:,:] = np.sqrt(alpha) * np.eye(nb_T)

    signal = np.concatenate([signal, np.zeros(nb_T)])
    return Ts, K, signal

def prepare_nnls_withoffset(tau, signal, min_T, max_T, nb_T, alpha):
    Ts = np.logspace(np.log10(min_T), np.log10(max_T), num=nb_T)
    nb_tau = tau.shape[0]
    K = np.zeros((nb_tau+nb_T,nb_T+1))
    for j in range(nb_T):
        K[:nb_tau,j] = kernel_T2(tau, Ts[j])


    K[nb_tau:,:-1] = np.sqrt(alpha) * np.eye(nb_T)
    K[:nb_tau,nb_T] = np.ones((nb_tau,))

    signal = np.concatenate([signal, np.zeros(nb_T)])
    return Ts, K, signal

def solve_nnls(K, signal):
    X, rnorm = nnls(K, signal)

    return X, rnorm

def fit(tau, signal, T_range, nb_T, alpha, allow_offset=False):
    if allow_offset:
        Ts, K, signal = prepare_nnls_withoffset(tau, signal, T_range[0], T_range[1], nb_T, alpha)
    else:
        Ts, K, signal = prepare_nnls(tau, signal, T_range[0], T_range[1], nb_T, alpha)

    X, rnorm = solve_nnls(K, signal)
    return Ts, X[:nb_T], rnorm

def prepare_fix_nnls(tau, signal, fix_ts, allow_offset=True):
    nb_T = len(fix_ts)
    nb_tau = tau.shape[0]
    if allow_offset:
        K = np.zeros((nb_tau, nb_T+1))
    else:
        K = np.zeros((nb_tau, nb_T))
    for j in range(nb_T):
        K[:nb_tau,j] = kernel_T2(tau, fix_ts[j])
    if allow_offset:
        K[:nb_tau,nb_T] = np.ones((nb_tau,))
    return K

def fit_fixcomponents(tau, signal, fix_Ts, allow_offset=False):
    nb_T = len(fix_Ts)
    K = prepare_fix_nnls(tau, signal, fix_Ts, allow_offset=allow_offset)

    X, rnorm = solve_nnls(K, signal)
    return X[:nb_T], rnorm

def get_data_dps(filepath):
    data = np.loadtxt(filepath, delimiter="\t")
    tau = data[:,1]*1e-3
    real = data[:,2]
    imag = data[:,3]

    return tau, real, imag

def remove_first_echo(tau, signal):
    return tau[1:], signal[1:]

def multi_components(x, As, Ts):
    asum = 0
    for A, T in zip(As, Ts):
        asum += A*np.exp(-x/T)
    return asum

def fit_four_components(taus, signal, Ts):
    def four_components(x, A1, A2, A3, A4):
        return multi_components(x, np.array([A1, A2, A3, A4]), Ts)

    popt, pcov = curve_fit(four_components, taus, signal)
    return popt, pcov

def fit_five_components(taus, signal, Ts):
    def five_components(x, A1, A2, A3, A4, A5):
        return multi_components(x, np.array([A1, A2, A3, A4, A5]), Ts)

    popt, pcov = curve_fit(five_components, taus, signal)
    return popt, pcov


if __name__ == "__main__":
    plt.figure()
    all_files=[]
    folder = "/data/georget/NMR/90d_erng/LGCMK_good_params_tauinit_025/"
    all_files.extend(glob(os.path.join(folder,"*.dps")))

    folder = "/data/georget/NMR/90d_erng/LGC_params_tau_init_025/"
    all_files.extend(glob(os.path.join(folder,"*.dps")))

    folder = "/data/georget/NMR/90d_erng/LGCMK_good_params_tauinit_025/"
    all_files.extend(glob(os.path.join(folder,"*.dps")))

    for file in all_files:
        tau, real, imag = get_data_dps(file)
        real, imag = phase(real, imag, 1, 3)

        tau, real = remove_first_echo(tau, real)
        plt.figure(1)
        plt.plot(tau, real)
        plt.xlim([0,5e-3])
        Ts, As, rnorm = fit(tau, real, [5e-5, 1],tau.shape[0], 10.0**-2)
        plt.figure(2)
        plt.semilogx(Ts, As)
#
#    signal = real
#    #tau, signal = remove_first_echo(tau, real)
#
#    plt.plot(tau, signal, ".")
#    plt.xlim([0,5])
#    plt.figure()
#    Ts, As, rnorm = fit(tau, signal, [1e-5, 1],tau.shape[0], 10.0**-7)
#    plt.semilogx(Ts, As)
#
#    for alpha in [-3,-4,-5,-6,-7]:
#        plt.figure()
#        for file in all_files:
#            tau, real, imag = get_data_dps(file)
#            real, imag = phase(real, imag, 1, 3)
#            tau, real  = remove_first_echo(tau, real)
#            Ts, As, rnorm = fit(tau, real, [5e-5, 1],tau.shape[0], 10.0**alpha)
#            plt.semilogx(Ts, As)
#        plt.title(r"$\alpha=10^{"+str(alpha)+r"}$")
#

#    file = all_files[4]
#    tau, real, imag = get_data_dps(file)
#    real, imag = phase(real, imag, 1, 3)
#    Ts, As, rnorm = fit(tau, real, [5e-5, 1],tau.shape[0], 10.0**-7)
#
#    peaks, prop = find_peaks(As)
#    plt.semilogx(Ts, As)
#    plt.semilogx([Ts[i] for i in peaks], [As[i] for i in peaks], "o")

    # ==========



#
#    alphas = np.linspace(-10,0,num=20)
#    norms = []
#    for alpha in alphas:
#        Ts, As, rnorm = fit(tau, real, [5e-5, 1], tau.shape[0], 10.0**alpha)
#        norms.append(rnorm)
#    plt.plot(alphas, norms, ".")


    #file = "20200129190038.dps"
    #folder = "/data/georget/NMR/S3-OH-L5B/S3-L5B-04_0075s/"
    #folder = "/data/georget/NMR/
#    all_files = glob(os.path.join(folder,"*.dps"))
#    for file in glob(os.path.join(folder,"*.dps")):
#        filepath = os.path.join(folder, file)
#    #file = "OPC-04-CPMG/OPC-04-CPMG_20181212161822.dps"
#
#        data = np.loadtxt(filepath, delimiter="\t")
#        tau = data[:,1]
#        real = data[:,2]
#        imag = data[:,3]
#
#    #plt.plot(tau, real, ".")
#    #plt.plot(tau, imag, ".")
#
#        real, imag = phase(real, imag, 0, 3)
#        plt.plot(tau, real, ".")
#
#        basename = os.path.splitext(file)[0]
#        with open(os.path.join(folder, "{0}_taus.txt".format(basename)), "w") as ofile:
#            for i in tau:
#                ofile.write("{0:.6f}\n".format(i))
#
#        with open(os.path.join(folder, "{0}_datas.txt".format(basename)), "w") as ofile:
#            for i in real:
#                ofile.write("{0:.6f}\n".format(i))
#
#    plt.xlim([0,5])
#        #plt.plot(tau, imag, ".")
#    #plt.plot(tau, imag, ".")


#




