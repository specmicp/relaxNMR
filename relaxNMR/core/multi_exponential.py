# Copyright (c) 2020 Fabien Georget <fabien.georget@epfl.ch>, EPFL
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import scipy.optimize as sco

"""Multi-exponential fitting."""

def single_exponential(tau, A, T):
    """A single exponential decay


    :param tau: the time
    :type tau: float or numpy array
    :param A: The magnitude (value at tau=0)
    :type A: float
    :param T: The relaxation time (T2)
    :type T: float
    """
    return A*np.exp(-tau/T)


def multi_exponential(signal, nb_exp):
    """Return a multi-exponential function to fit.

    :param signal: The signal to fit
    :type signal: :class:relaxNMR:core:signal:Signal
    :param nb_exp: the number of components (i.e. number of discrete relaxation times)
    :type nb_exp: int
    """
    def to_fit(x):
        sol = 0
        for i in range(nb_exp):
            sol += single_exponential(signal.tau, x[2*i], x[2*i+1])
        return signal.signal-sol
    return to_fit


def fit_multi_exponential(signal, nb_exp, T_bounds):
    """Fit the signal to a multi-exponential.

    :type signal: the signal to fit
    :type signal: :class:relaxNMR:core:signal:Signal
    :param nb_exp: the number of components (i.e. number of discrete relaxation times)
    :type nb_exp: int
    :param T_bounds: The bounds for the relaxation times of each component
    :type T_bounds: sequence of 2-tuples (min_T, max_T)
    """
    bounds_inf = np.zeros((nb_exp*2,), dtype=np.float64)
    bounds_sup = np.zeros((nb_exp*2,), dtype=np.float64)
    x0 = np.zeros((nb_exp*2,), dtype=np.float64)
    for i in range(nb_exp):
        bounds_inf[i*2] = 0.0
        bounds_sup[i*2] = np.inf
        bounds_inf[i*2+1] = T_bounds[i][0]
        bounds_sup[i*2+1] = T_bounds[i][1]
        x0[i*2] = 0.001
        x0[i*2+1] = T_bounds[i][0] # -T_bounds[i][0])/2
    func = multi_exponential(signal, nb_exp)
    res = sco.least_squares(func, x0, bounds=(bounds_inf, bounds_sup))

    sol = res.x
    As = np.zeros((nb_exp,))
    Ts = np.zeros((nb_exp,))
    for i in range(nb_exp):
        As[i] = sol[2*i]
        Ts[i] = sol[2*i+1]

    return As, Ts, res


def fit_multi_exponential_collection(collection, nb_exp, T_bounds):
    As_all = np.zeros((len(collection), nb_exp))
    Ts_all = np.zeros((len(collection), nb_exp))

    for ind, s in enumerate(collection):
        As, Ts, res = fit_multi_exponential(s, nb_exp, T_bounds)
        As_all[ind, :] = As
        Ts_all[ind, :] = Ts

    return As_all, Ts_all

def multi_exponential_fixT(signal, nb_exp, Ts):
    """Return a multi-exponential function with fixed relaxation times to fit.

    :param signal: The signal to fit
    :type signal: :class:relaxNMR:core:signal:Signal
    :param nb_exp: the number of components (i.e. number of discrete relaxation times)
    :type nb_exp: int
    :param Ts: the fixed relaxation times
    :type Ts: array-like
    """

    def to_fit(x):
        sol = 0
        for i in range(nb_exp):
            sol += single_exponential(signal.tau, x[i], Ts[i])
        return signal.signal-sol
    return to_fit


def fit_multi_exponential_fixT(signal, nb_exp, Ts):
    """Fit the signal to a multi-exponential with fix relaxation time.


    :type signal: the signal to fit
    :type signal: :class:relaxNMR:core:signal:Signal
    :param nb_exp: the number of components (i.e. number of discrete relaxation times)
    :type nb_exp: int
    :param Ts: the fixed relaxation times
    :type Ts: array-like
    """
    bounds_inf = np.zeros((nb_exp,), dtype=np.float64)
    bounds_sup = np.zeros((nb_exp,), dtype=np.float64)
    x0 = np.zeros((nb_exp,), dtype=np.float64)
    for i in range(nb_exp):
        bounds_inf[i] = 0.0
        bounds_sup[i] = np.inf
        x0[i] = 0.001
    func = multi_exponential_fixT(signal, nb_exp, Ts)
    res = sco.least_squares(func, x0, bounds=(bounds_inf, bounds_sup))
    As = res.x
    return As, res
