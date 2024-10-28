bt_ocean
========

bt_ocean is a differentiable pseudospectral solver for the barotropic vorticity
equation on a beta-plane, for classic wind-forced barotropic ocean gyre
simulations.

bt_ocean is designed to be simple, lightweight, and fast on a single GPU. The
aim is enable rapid testing of ocean-relevant machine learning techniques in a
problem with multiple flow regimes, boundary effects, and eddy energy
backscatter.

Features
--------

- A pseudospectral solver for the 2D barotropic vorticity equation in a
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

bt_ocean numerics
-----------------

bt_ocean makes use of the following numerical methods.

- A tensor product Chebyshev pseudospectral discretization in a rectangular
  domain, using Chebyshev nodes of the second kind.
- A CNAB2 time discretization, started with a CNAB1 step. Elliptic problems
  associated with potential vorticity inversion and the implicit timestep are
  solved using Kronecker product based direct linear solvers [1]_. Dealiasing,
  with a dealising factor of two, is applied for the non-linear advection term.
- No-normal-flow and free slip boundary conditions, applied via homogeneous
  Dirichlet boundary conditions for both the stream function and vorticity.

Known limitations
-----------------

- bt_ocean uses Kronecker product based direct linear solvers [1]_. However JAX
  is currently unable to differentiate with respect to the solution to an
  eigenproblem (specifically with respect to the eigenvectors). This means for
  example that it is not currently possible to differentiate with respect to
  parameters appearing in these solvers, such as the linear bottom drag and
  Laplacian viscosity coefficients.

.. [1] Using equation (3.9) of https://doi.org/10.1007/BF01386067.

Indices
-------

* :ref:`modindex`
