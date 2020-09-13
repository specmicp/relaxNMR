relaxNMR
========


A simple package to analyse 1D ^1^H NMR relaxometry experiments in cement paste. It was developed to allow batch analysis of experiments
From the raw signals, the inverse laplace transform (ILT) is computed to obtain the relaxation times.

A non-negative least square algorithm with Tikhonov regularization is used for the ILT.

Minimal example
---------------

~~~python
    # preprocessing
signals = read_folder("mydata/", ".dps")
signals.phase(1,3)
signals.remove_first_echo()
average_signal = signals.average()

Trange=(1e-5, 1) # the range of relaxation times
alpha = 1e-2     # the penalization parameter

    # fit individual experiment
fitted = ILT_fit_collection(signals, Trange, alpha)
fitted.normalize()

   # fit average
fitted_average = ILT_fit(average_signal, Trange, alpha)
fitted_average.normalize()
~~~

About
-----

Developed by Fabien Georget, at the [Laboratory of Construction Materials](https://www.epfl.ch/labs/lmc/), EPFL, Lausanne, Switzerland
