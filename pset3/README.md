This directory contains code for the FDTD simulation. It is constructed in a way
that will minimize difficulties in making this repository an installable package.
For the time being, some hand-holding is needed in order to make it work. Basic
instructions for use are detailed below.

## Dependencies ##
* ``numpy``
* ``astropy``
* ``cached_property``

## Script Usage ##
The scripts used for testing this code and running simulations are located in the
``fdtd/scripts`` directory. Assuming you have installed the required dependencies
and cloned the contents of this repository, you can run one of the scripts by
navigating to the ``fdtd/scripts`` directory and running, e.g.
```
    python test_wavelet.py
```
Feel free to modify the contents of the scripts if you would like to experiment.
