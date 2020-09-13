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

"""The inverse Laplace transform"""

import numpy as np

from scipy.optimize import nnls

from relaxNMR.core.signal import ComplexSignal, MagnitudeSignal,\
     FittedSignal, FittedSignalCollection


def kernel_T2(tau, T):
    """The T2 exponential decay"""
    return np.exp(-tau/T)


def logarithm_Tspace(Trange, nb_T):
    """Generate a set of relaxation times logarithmically spaced.

    :param Trange: minimum and maximum
    :type Trange: 2-tuple
    :param nb_T: the number of discrete relaxation times, <= number of times in the signal
    :type nb_T: int
    """
    return np.logspace(np.log10(Trange[0]), np.log10(Trange[1]), num=nb_T)


def ILT_fit_collection(collection, Trange, alpha, nb_T=None, kernel=kernel_T2):
    """Fit a collection usingin the inverse Laplace transform.


    :param collection: a collection of NMR signals
    :type collection: :class:relaxNMR:core:signal:SignalCollection
    :param Trange: Minimum and maximum relaxation times
    :type Trange: 2-tuple
    :param alpha: Regularisation parameter
    :type alpha: float
    :param nb_T: the number of discrete relaxation times, <= number of times in the signal
    :type nb_T: int
    :param kernel: the kernel of the Fredholm equation
    :type kernel: function

    """
    fitter = ILTFitter1D(Trange, alpha, collection[0], nb_T, kernel)
    fitted_signals = FittedSignalCollection()
    for signal in collection:
        fit, rnorm = fitter.invert(signal)
        fitted_signals.append(fit)
    return fitted_signals

def ILT_fit(signal, Trange, alpha, nb_T=None, kernel=kernel_T2):
    fitter = ILTFitter1D(Trange, alpha, signal, nb_T, kernel)
    fit, _ = fitter.invert(signal)
    return fit

class ILTFitter1D:
    """Use this class to do a 1D inverse laplace transform.

    Use non-negative constraint and Tikhinov regularization."""
    def __init__(self, Trange, alpha, signal, nb_T=None, kernel=kernel_T2):
        """
        Create a ILT fitter. This instance will be valid for any signal with
        the same type and the same times as the provided signal.

        :param Trange: Minimum and maximum relaxation times
        :type Trange: 2-tuple
        :param alpha: Regularisation parameter
        :type alpha: float
        :param signal: One of the signal to invert
        :type signal: :class:relaxNMR:core:signal:Signal
        :param nb_T: the number of discrete relaxation times, <= number of times in the signal
        :type nb_T: int
        :param kernel: the kernel of the Fredholm equation
        :type kernel: function

        """
        self.nb_tau = signal.size
        self.tau = signal.tau

        if nb_T is None:
            self.nb_T = self.nb_tau
        else:
            if nb_T > self.nb_tau:
                raise ValueError("The number of T2s specified is bigger than the signal")
            self.nb_T = nb_T

        if isinstance(signal, ComplexSignal):
            self.has_offset = False
        else:
            self.has_offset = True

        self.Ts = logarithm_Tspace(Trange, self.nb_T)
        self.alpha = alpha
        self._set_kernel(signal.tau)

    def _set_kernel(self, tau):
        """Create the kernel used in the fitting."""
        nb_T = self.nb_T
        nb_tau = self.nb_tau

        if self.has_offset:
            K = np.zeros((nb_tau+nb_T, nb_T+1))
        else:
            K = np.zeros((nb_tau+nb_T, nb_T))

        for j in range(nb_T):
            K[:nb_tau, j] = kernel_T2(tau, self.Ts[j])

        # regularization
        K[nb_tau:, :nb_T] = np.sqrt(self.alpha) * np.eye(nb_T)
        if self.has_offset:
            K[:nb_tau, nb_T] = np.ones((nb_tau, ))

        self.K = K

    def invert(self, signal):
        """
        Invert the signal

        Assume that signal is similar to the signal use to create the filter
        (i.e. similar class, similar size)
        """
        assert(signal.size == self.nb_tau)
        rhs = np.concatenate([signal.signal, np.zeros(self.nb_T)])

        X, rnorm = nnls(self.K, rhs)
        As = X[:self.nb_T]
        if self.has_offset:
            offset = X[self.nb_T]
        else:
            offset = 0.0

        return FittedSignal(self.tau, As, self.K, self.Ts, offset=offset), rnorm
