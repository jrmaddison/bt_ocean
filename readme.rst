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
