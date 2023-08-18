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

"""
Base classes to represent experiments
"""

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["ComplexSignal", "MagnitudeSignal", "FittedSignal",
           "SignalCollection",
           "ComplexSignalCollection", "MagnitudeSignalCollection",
           "FittedSignalCollection", "ProfileSignalCollection"]


class Signal(ABC):
    def __init__(self, tau):
        self._tau = tau

    @property
    def tau(self):
        return self._tau

    @property
    def size(self):
        return self._tau.shape[0]

    @abstractmethod
    def _signal_impl(self):
        pass

    @property
    def signal(self):
        return self._signal_impl()


class ComplexSignal(Signal):
    """A complex signal"""

    def __init__(self, tau, real, imag):
        assert (real.shape[0] == tau.shape[0])
        assert (imag.shape[0] == tau.shape[0])

        super().__init__(tau)
        self._real = real
        self._imag = imag

    @property
    def real(self):
        return self._real

    @property
    def imag(self):
        return self._imag

    def is_complex(self):
        return True

    def is_magnitude(self):
        return False

    def _signal_impl(self):
        return self._real

    def phase(self, start_echo, nb_echo):
        thetas = []
        for i in np.arange(start_echo, start_echo + nb_echo+1):
            if self.real[i] < 0:
                thetas.append(np.arctan(self.imag[i] / self.real[i]) + np.pi)
            else:
                thetas.append(np.arctan(self.imag[i] / self.real[i]))

        theta = np.array(thetas).mean()

        real1 = np.cos(theta)*self.real + np.sin(theta)*self.imag
        imag1 = -np.sin(theta)*self.real + np.cos(theta)*self.imag

        self._real = real1
        self._imag = imag1

    def magnitude(self):
        signal = np.sqrt(np.power(self._real, 2) + np.power(self._imag, 2))
        return MagnitudeSignal(self.tau, signal)

    def remove_first_echo(self):
        self._tau = self._tau[1:]
        self._real = self._real[1:]
        self._imag = self._imag[1:]


class MagnitudeSignal(Signal):
    """A signal in magnitude mode"""

    def __init__(self, tau, signal):
        assert (signal.shape[0] == tau.shape[0])

        super().__init__(tau)
        self._signal = signal

    def is_complex(self):
        return False

    def is_magnitude(self):
        return True

    def _signal_impl(self):
        return self._signal

    def remove_first_echo(self):
        self._tau = self._tau[1:]
        self._signal = self._signal[1:]


class FittedSignal(Signal):
    """A fitted signal"""

    def __init__(self, tau, magnitudes, kernel, Ts, offset=0.0):
        super().__init__(tau)
        self._kernel = kernel
        self._As = np.asarray(magnitudes)
        self._offset = offset
        self._Ts = Ts

    def _signal_impl(self):
        return np.dot(self._kernel[:self.size, :self._As.shape[0]], self._As) + self._offset

    @property
    def offset(self):
        """The constant offset of the signal."""
        return self._offset

    @property
    def magnitude(self):
        """The magnitudes for each relaxation time."""
        return self._As

    @property
    def size_Ts(self):
        """The number of relaxation times."""
        return self._Ts.shape[0]

    @property
    def Ts(self):
        """The relaxation times."""
        return self._Ts

    @property
    def logTs(self):
        """The log10 relaxtion times"""
        return [np.log10(t) for t in self._Ts]

    def normalize(self):
        """Normalize the magnitudes."""
        self._As = self._As / np.sum(self._As)


class SignalCollection:
    """A container of signals. Assume all signals have the same size"""

    def __init__(self, signals=None):
        """
        :param signals: Signals to initialize the container
        :type signals: list-like container
        """
        self._signals = []
        if signals is not None:
            self._signals.extend(signals)

    def __len__(self):
        return len(self._signals)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return SignalCollection(signals=self._signals[index])
        return self._signals[index]

    @property
    def shape(self):
        return (self.__len__(), self.size)

    @property
    def size(self):
        return self._signals[0].size

    @property
    def tau(self):
        return self._signals[0].tau

    def append(self, signal):
        """Add a signal to the collection"""
        assert (isinstance(signal, Signal))
        self._signals.append(signal)

    def extend(self, signals):
        """Append all signals to this container."""
        if isinstance(signals, SignalCollection):
            self._signals.extend(signals._signals)
        else:
            self._signals.extend(signals)

    def __iter__(self):
        return self._signals.__iter__()

    def asarray(self):
        """Return the signals as arrays

        """
        return NotImplementedError("Use a subclass to precise the nature of the scans")

    def is_single_class(self, cls):
        """Return True if all signals are instances fo 'cls'"""
        for signal in self._signals:
            if not isinstance(signal, cls):
                return False
        return True

    def is_complex(self):
        """Return true if the signal is complex"""
        return self._is_single_class(ComplexSignal)

    def is_magnitude(self):
        """Return true if the signal is in magnitude mode"""
        return self._is_single_class(MagnitudeSignal)

    def is_fitted(self):
        """Return true if this signal is a reconstructed signal."""
        return self._is_single_class(FittedSignal)

    def average(self):
        """Returns an average of the collection as a single signal."""
        return NotImplementedError("Use a subclass to precise the nature of the scans")

    def average_byN(self, n):
        """Average each n signals together to form a new collection"""
        return NotImplementedError("Use a subclass to precise the nature of the scans")


def average_complex_signals(signals_array):
    size = signals_array[0].size
    alen = len(signals_array)

    average_real = np.zeros((size,))
    average_imag = np.zeros((size,))
    for s in signals_array:
        average_real += s.real
        average_imag += s.imag
    average_real /= alen
    average_imag /= alen
    return ComplexSignal(signals_array[0].tau, average_real, average_imag)


class ComplexSignalCollection(SignalCollection):
    """ A container of complex signals"""

    def __init__(self, signals=None):
        super().__init__(signals)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return ComplexSignalCollection(signals=self._signals[index])
        return self._signals[index]

    def append(self, signal):
        assert (isinstance(signal, ComplexSignal))
        self._signals.append(signal)

    def asarray(self):
        """Return the real and imaginary parts of all the signals as arrays"""
        real = np.zeros(self.shape)
        imag = np.zeros(self.shape)
        for ind, s in enumerate(self._signals):
            real[ind, :] = s.real
            imag[ind, :] = s.imag
        return real, imag

    def is_complex(self):
        """Return true if the signal is complex"""
        return True

    def is_magnitude(self):
        """Return true if the signal is in magnitude mode"""
        return False

    def is_fitted(self):
        """Return true if this signal is a reconstructed signal."""
        return True

    def average(self):
        """Returns an average of the collection as a single signal."""
        return average_complex_signals(self._signals)

    def average_byN(self, n):
        """Average each n signals together to form a new collection"""
        new_signals = ComplexSignalCollection()
        for i in range(int(np.floor(len(self._signals)/n))):
            new_signals.append(average_complex_signals(
                self._signals[n*i:n*(i+1)]))
        return new_signals

    def phase(self, start_echo, nb_echo):
        """Phase all the signals"""
        for s in self._signals:
            s.phase(start_echo, nb_echo)

    def remove_first_echo(self):
        """Remove first echos from all the signals"""
        for s in self._signals:
            s.remove_first_echo()


def average_magnitude_signal(signals_array):
    size = signals_array[0].size
    alen = len(signals_array)

    average_signal = np.zeros((size,))
    for s in signals_array:
        average_signal += s.signal
    average_signal /= alen
    return MagnitudeSignal(signals_array[0].tau, average_signal)


class MagnitudeSignalCollection(SignalCollection):
    def __init__(self, signals=None):
        super().__init__(signals)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return MagnitudeSignalCollection(signals=self._signals[index])
        return self._signals[index]

    def append(self, signal):
        assert (isinstance(signal, MagnitudeSignal))
        self._signals.append(signal)

    def asarray(self):
        """Return an array containing the values of all the signals"""
        signal = np.zeros(self.shape)
        for ind, s in enumerate(self._signals):
            signal[ind, :] = s.signal
        return signal

    def is_complex(self):
        """Return true if the signal is complex"""
        return False

    def is_magnitude(self):
        """Return true if the signal is in magnitude mode"""
        return True

    def is_fitted(self):
        """Return true if this signal is a reconstructed signal."""
        return False

    def average(self):
        """Returns an average of the collection as a single signal."""
        return average_magnitude_signal(self._signals)

    def average_byN(self, n):
        """Average each n signals together to form a new collection"""
        new_signals = MagnitudeSignalCollection()
        for i in range(int(np.floor(len(self._signals)/n))):
            new_signals.append(average_magnitude_signal(
                self._signals[n*i:n*(i+1)]))
        return new_signals


class FittedSignalCollection(SignalCollection):

    def __init__(self, signals=None):
        super().__init__(signals)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return FittedSignalCollection(signals=self._signals[index])
        return self._signals[index]

    def append(self, signal):
        assert (isinstance(signal, FittedSignal))
        self._signals.append(signal)

    def asarray(self):
        """Return an array containing the values of all the signals"""
        shape = (self.__len__(), self._signals[0].size_Ts)
        shape_offset = (self.__len__(), )
        magnitude = np.zeros(shape)
        offset = np.zeros(shape_offset)
        for ind, s in enumerate(self._signals):
            magnitude[ind, :] = s.magnitude
            offset[ind] = s.offset
        return magnitude, offset

    def is_complex(self):
        """Return true if the signal is complex"""
        return False

    def is_magnitude(self):
        """Return true if the signal is in magnitude mode"""
        return False

    def is_fitted(self):
        """Return true if this signal is a reconstructed signal."""
        return True

    def average(self):
        """Returns an average of the collection as a single signal."""
        average_magnitudes = np.zeros_like(self._signals[0].magnitude)
        average_offset = 0.0
        for s in self._signals:
            average_magnitudes += s.magnitude
            average_offset += s.offset
        average_magnitudes /= self.__len__()
        average_offset /= self.__len__()
        return FittedSignal(self.tau, average_magnitudes,
                            self._signals[0].kernel,
                            self._signals[0].Ts, offset=average_offset)

    def average_byN(self, n):
        """Average each n signals together to form a new collection"""
        new_signals = FittedSignalCollection()
        for i in range(int(np.floor(len(self._signals)/n))):
            average_magnitudes = np.zeros_like(self._signals[0].magnitude)
            average_offset = 0.0
            for s in self._signals[n*i:n*(i+1)]:
                average_magnitudes += s.magnitude
                average_offset += s.offset
            average_magnitudes /= self.__len__()
            average_offset /= self.__len__()
            new_signals.append(
                FittedSignal(self.tau, average_magnitudes,
                             self._signals[0].kernel, self._signals[0].Ts,
                             offset=average_offset))
        return new_signals

    def normalize(self):
        """Normalize the magnitudes."""
        for s in self._signals:
            s.normalize()


class ProfileSignalCollection:
    def __init__(self, depths, signals):
        self._depths = depths
        if not hasattr(signals, "__len__"):
            raise ValueError("signals must be a list-like of signals")
        self._signals = signals

    def is_complex(self):
        """Return true if the signal is complex"""
        return self._signals[0].is_complex()

    def is_magnitude(self):
        """Return true if the signal is in magnitude mode"""
        return self._signals[0].is_magnitude()

    def is_fitted(self):
        """Return true if this signal is a reconstructed signal."""
        return self._signals[0].is_fitted()

    def echos_correction(self, cor1, cor2):
        for s in self._signals:
            s.real[0] /= cor1
            s.imag[1] /= cor2

    def at(self, depth):
        ind = np.where(self._depths >= depth)[0]
        return self._signals[ind]

    def depths(self):
        return self._depths

    def __getitem__(self, int_or_slice):
        if isinstance(int_or_slice, int):
            return self._signals[int_or_slice]
        else:
            return ProfileSignalCollection(
                self._depths[int_or_slice],
                self._signals[int_or_slice])

    def items(self):
        for i, d in enumerate(self._depths):
            yield d, self._signals[i]

    def _get_average_f(self):
        if self.is_complex():
            fn_average = average_complex_signals
        elif self.is_magnitude():
            fn_average = average_magnitude_signal
        else:
            raise RuntimeError("Unknow signal")
        return fn_average

    def average(self):
        fn_average = self._get_average_f()
        return fn_average(self._signals)

    def average_depth(self, depth_start, depth_end):
        inds = np.where(self._depths >= depth_start)[0][0]
        inde = np.where(self._depths >= depth_end)[0]
        if inde.size == 0:
            inde = None
        else:
            inde = inde[0]

        fn_average = self._get_average_f()
        return fn_average(self._signals[inds:inde])
