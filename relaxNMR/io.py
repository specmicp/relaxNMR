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
    MagnitudeSignalCollection, ComplexSignalCollection, \
    ProfileSignalCollection, QuadEchoCollection


class SignalFormat:
    """Format of the CSV file"""

    def __init__(self,
                 delimiter,
                 time_col=0,
                 time_factor=1.0,
                 real_col=None, imag_col=None,
                 magn_col=None,
                 skiprows=0):
        self.delimiter = delimiter
        self.time_col = time_col
        self.time_factor = time_factor
        self.skiprows = skiprows
        self.autostrip = False

        self._is_magnitude = (magn_col is not None)

        if self._is_magnitude:
            self.real_col = None
            self.imag_col = None
            self.magn_col = magn_col

        else:
            self.magn_col = None
            if real_col is None:
                raise ValueError(
                    "Either real_col or magn_col must be provided")
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

MQC_txt_format = SignalFormat(" ",
                              time_col=0, time_factor=1e-6,
                              real_col=4, imag_col=8
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

    # data = np.loadtxt(filepath, delimiter=fmt.delimiter, skiprows=fmt.skiprows)
    data = np.genfromtxt(filepath, delimiter=fmt.delimiter,
                         skip_header=fmt.skiprows,
                         autostrip=fmt.autostrip)

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


def read_mouse_signal(filepath, delimiter="\t"):
    """Read a signal from a mouse file

    :param filepath: the path to a mouse ".CPMG"/".CPMGACQP" file
    :type filepath: str
    """
    ext = os.path.splitext(filepath)[1]
    data = np.loadtxt(filepath, delimiter=delimiter)
    depths = np.unique(data[:, 0])
    signals = {}
    if ext == ".cpmgacqp":
        for d in depths:
            subview = data[data[:, 0] == d, :]
            signals[d] = ComplexSignal(
                subview[:, 2]*1e-3, subview[:, 4], subview[:, 5])
    elif ext == ".cpmg":
        for d in depths:
            subview = data[data[:, 0] == d, :]
            signals[d] = MagnitudeSignal(subview[:, 2]*1e-3, subview[:, 4])
    return depths, signals


def read_mouse_profile(filepath, delimiter="\t"):
    """Read signal from a depth profiles obtained from the NMR Mouse"""
    ext = os.path.splitext(filepath)[1]
    data = np.loadtxt(filepath, delimiter=delimiter)
    depths = np.unique(data[:, 0])

    if ext == ".dat":
        # Mouse PM25 - decays in two files: -decays.dat, and -decaysimag.dat
        base = os.path.splitext(filepath)[0]
        real_files = base+"-decays.dat"
        imag_files = base+"-decaysimag.dat"
        depth_file = base+".dat"

        real_data = np.loadtxt(real_files, delimiter=delimiter)
        imag_data = np.loadtxt(imag_files, delimiter=delimiter)

        depths = np.loadtxt(depth_file, delimiter=delimiter)[:, 0]

        signals = []
        for ind in range(len(depths)):
            signals.append(ComplexSignal(
                real_data[:, 0]/1000, real_data[:, ind+1], imag_data[:, ind+1]))
    elif ext in [".cpmg", ".cpmgacqp"]:
        data = np.loadtxt(filepath, delimiter=delimiter)
        depths = np.unique(data[:, 0])
        signals = []

        if ext == ".cpmgacqp":
            for d in depths:
                subview = data[data[:, 0] == d, :]
                signals.append(ComplexSignal(
                    subview[:, 2]/1000, subview[:, 4], subview[:, 5]))
        elif ext == ".cpmg":
            for d in depths:
                subview = data[data[:, 0] == d, :]
                signals.append(MagnitudeSignal(
                    subview[:, 2]/1000, subview[:, 4]))

    return ProfileSignalCollection(depths, signals)


def read_delay_mqc(filepath):
    parameters = filepath.replace(".Dat.txt", ".Par.txt")
    with open(parameters, "r") as ofile:
        for line in ofile:
            param, value = line.split()
            if param == "D1":
                return float(value)
            else:
                continue
    return ValueError("Delay not found in parameter file: {0}.".format(parameters))


def read_quadecho_folder(folder, fmt, extension, tau_f):
    """

    Parameters
    ----------
    folder : str
        The folder to read
    fmt: SignalFormat
        The format used to read the signal
    extension: str:
        The extension of files containing individual signals
    tau_f : str
        The function to read the tau delay, takes the signal filename as input

    Returns
    -------
    A QuadEchoCollection

    """
    if not extension.startswith("."):
        extension = "." + extension
    pattern = "*" + extension
    all_files = glob(os.path.join(folder, pattern))
    all_files.sort(key=os.path.getmtime)

    collection = QuadEchoCollection()
    for file in all_files:
        delay = tau_f(file)
        signal = read_signal(file, fmt)
        collection[delay] = signal

    return collection


def read_quadecho_mqc(folder):
    """A simple wrapper around read_quadecho_folder for the MQC equipment."""
    return read_quadecho_folder(folder,
                                fmt=MQC_txt_format,
                                extension=".Dat.txt",
                                tau_f=read_delay_mqc)
