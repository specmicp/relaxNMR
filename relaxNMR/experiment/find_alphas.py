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

from relaxNMR.core.ilt import ILTFitter1D
from relaxNMR.core.signal import Signal, SignalCollection


__all__ = ["rnorm_for_alphas"]

def rnorm_for_alphas(signal, alphas, Ts_range):
    """Return the norm of residuals for a set of regularization parameter.

    :param signal: a NMR signal or a collection of NMR signals
    :type signal: :class:relaxNMR:core:signal:Signal or :class:relaxNMR:core:signal:SignalCollection
    :param alphas: The regularization parameter to test
    :type alphas: array like of float
    :returns: a vector (signal) or an array (signalxresiduals) containing the residuals norms
    """
    if isinstance(signal, Signal):
        return rnorm_for_alphas_signal(signal, alphas, Ts_range)
    elif isinstance(signal, SignalCollection):
        return rnorm_for_alphas_collection(signal, alphas, Ts_range)


def rnorm_for_alphas_signal(signal, alphas, Ts_range):
    """Return the norm of residuals for a set of regularization parameter for a signal.

    :param signal: a NMR signal
    :type signal: :class:relaxNMR:core:signal:Signal
    :param alphas: The regularization parameter to test
    :type alphas: array like of float
    :returns:  a vector (signal) containing the residuals norms
    """
    rnorms = np.zeros((len(alphas),))
    for j, alpha in enumerate(alphas):
        inverter = ILTFitter1D((1e-5, 1.0), alpha, signal)
        rnorms[j] = inverter.invert(signal)[1]
    return rnorms


def rnorm_for_alphas_collection(signals, alphas, Ts_range):
    """Return the norm of residuals for a set of regularization parameter for a signal.

    :param signal: a collection of NMR signals
    :type signal: :class:relaxNMR:core:signal:SignalCollection
    :param alphas: The regularization parameter to test
    :type alphas: array like of float
    :returns:  an array (signalxresiduals) containing the residuals norms
    """
    rnorms = np.zeros((len(signals), len(alphas)))
    for ind, s in enumerate(signals):
        for j, alpha in enumerate(alphas):
            inverter = ILTFitter1D((1e-5, 1.0), alpha, s)
            rnorms[ind, j] = inverter.invert(s)[1]
    return rnorms
