bt_ocean
========

bt_ocean is a differentiable finite difference solver for the barotropic
vorticity equation on a beta-plane, for classic wind-forced barotropic ocean
gyre simulations.

bt_ocean is designed to be simple, lightweight, and fast on a single GPU. The
aim is enable rapid testing of ocean-relevant machine learning techniques in a
problem with multiple flow regimes, boundary effects, and eddy energy
backscatter.

Features
--------

- A finite difference solver for the 2D barotropic vorticity equation in a
  rectangular domain, for simulations of classic wind-driven Munk-Stommel ocean
  gyre problems.
- Uses the JAX library, providing GPU and autodiff support.
- Integrates with the Keras library for online training of neural networks.

Examples
--------

The following Jupyter notebooks introduce bt_ocean.

- `Getting started with bt_ocean <examples/0_getting_started.ipynb>`__:
  Introduces the configuration and running of bt_ocean, and reverse mode
  autodiff of a diagnostic.
- `Keras integration <examples/1_keras_integration.ipynb>`__: Combining
  bt_ocean with Keras. Describes the key building blocks which can be used to
  apply bt_ocean for online learning.
- `Steady state problems <examples/2_steady_state.ipynb>`__: Implicit
  differentiation for steady-state problems.

Source
------

The source code is available from the
`bt_ocean GitHub repository <https://github.com/jrmaddison/bt_ocean>`_.

Indices
-------

* :ref:`modindex`
