
fMRI Alignment Benchmark
========================
|Build Status| |Python 3.11+|



Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to the resolution of the functional brain alignment problem.
Specifically, we evaluate the decoding accuracy of a classifier on a left-out subject after 
training it on data from other subjects, following functional alignment using the methods detailed below.
We benchmark the following methods, all implemented in the [fmralign](https://github.com/Parietal-INRIA/fmralign/) package:

* Piecewise Procrustes
* Ridge Regression
* FastSRM
* Optimal Transport
* FUGW
* Hyperalignment : 
  - pcha : Parcellation Hyperalignment
  - slha : Searchlight Hyperalignment

on five different datasets:

* IBC RSVP
* IBC Sounds
* Courtois Neuromod
* BOLD5000

Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/pbarbarant/fmralign_benchmark_old
   $ benchopt run fmralign_benchmark_old

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run fmralign_benchmark_old -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/pbarbarant/fmralign_benchmark_old/workflows/Tests/badge.svg
   :target: https://github.com/pbarbarant/fmralign_benchmark_old/actions
.. |Python 3.11+| image:: https://img.shields.io/badge/python-3.11%2B-blue
   :target: https://www.python.org/downloads/release/python-3115/
