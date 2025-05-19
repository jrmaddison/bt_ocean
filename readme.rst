bt_ocean
========
|tests_32| |tests_64| |license|

.. |tests_32| image:: https://github.com/jrmaddison/bt_ocean/actions/workflows/test-bt_ocean-32.yml/badge.svg?branch=main&event=push
    :alt: 32-bit tests
    :target: https://github.com/jrmaddison/bt_ocean/actions/workflows/test-bt_ocean-32.yml

.. |tests_64| image:: https://github.com/jrmaddison/bt_ocean/actions/workflows/test-bt_ocean-64.yml/badge.svg?branch=main&event=push
    :alt: 64-bit tests
    :target: https://github.com/jrmaddison/bt_ocean/actions/workflows/test-bt_ocean-64.yml

.. |license| image:: https://img.shields.io/badge/license-MIT-green?style=flat-square
   :alt: license
   :target: https://github.com/jrmaddison/bt_ocean/blob/main/LICENSE

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

Documentation
-------------

- `bt_ocean documentation <https://jrmaddison.github.io/bt_ocean>`_
- `Module index <https://jrmaddison.github.io/bt_ocean/py-modindex.html>`_
