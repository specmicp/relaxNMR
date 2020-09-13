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
Read/Write signals
"""

import os.path
from glob import glob

import numpy as np
from relaxNMR.core import MagnitudeSignal, ComplexSignal,\
    MagnitudeSignalCollection, ComplexSignalCollection


class SignalFormat:
    """Format of the CSV file"""
    def __init__(self,
                 delimiter,
                 time_col=0,
                 time_factor=1.0,
                 real_col=None, imag_col=None,
                 magn_col=None):
        self.delimiter = delimiter
        self.time_col = time_col
        self.time_factor = time_factor

        self._is_magnitude = (magn_col is not None)

        if self._is_magnitude:
            self.real_col = None
            self.imag_col = None
            self.magn_col = magn_col

        else:
            self.magn_col = None
            if real_col is None:
                raise ValueError("Either real_col or magn_col must be provided")
            self.real_col = real_col
            self.imag_col = imag_col

    @property
    def is_magnitude(self):
        return self._is_magnitude

# The default format
default_format = SignalFormat("\t",
                              time_col=1, time_factor=1e-3,
                              real_col=2, imag_col=3
                              )


def read_signal(filepath, fmt=None):
    """Read a signal from a file.


    :param filepath: the path to the csv-like file
    :type filepath: str
    :param fmt: Format of the file
    :type fmt: :class:SignalFormat
    """
    if fmt is None:
        fmt = default_format

    data = np.loadtxt(filepath, delimiter=fmt.delimiter)

    tau = data[:, fmt.time_col]*fmt.time_factor
    if fmt.is_magnitude:
        signal = data[:, fmt.magn_col]
        return MagnitudeSignal(tau, signal)
    else:
        real = data[:, fmt.real_col]
        if fmt.imag_col is None:
            raise Warning("No imaginary column, assuming 0")
            imag = np.zeros_like(real)
        else:
            imag = data[:, fmt.imag_col]
        return ComplexSignal(tau, real, imag)


def read_folder(path, extension, fmt=None):
    """Read all signals in a folder

    :param path: The path to the folder
    :type path: str
    :param extension: The extension of the file (e.g. dps)
    :type extension: str
    """
    if not extension.startswith("."):
        extension = "." + extension
    pattern = "*" + extension
    all_files = glob(os.path.join(path, pattern))
    all_files.sort(key=os.path.getmtime)

    if fmt is None:
        fmt = default_format

    if fmt.is_magnitude:
        collection = MagnitudeSignalCollection()
    else:
        collection = ComplexSignalCollection()
    for file in all_files:
        signal = read_signal(file, fmt=fmt)
        collection.append(signal)
    return collection
