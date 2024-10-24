"""A set of base model parameters. Physical parameters (excluding the viscous
parameter and density) are from

    - 'Parameterization of ocean eddies: Potential vorticity mixing, energetics
      and Arnold's first stability theorem', David P. Marshall and Alistair J.
      Adcroft, Ocean Modelling 32(3-4), pp. 188--204, 2010,
      doi: 10.1016/j.ocemod.2010.02.001
"""

import jax.numpy as jnp

from .model import Parameters

__all__ = \
    [
        "n_hour",
        "n_day",
        "n_year",

        "parameters",

        "rho_0",
        "D",
        "tau_0",
        "Q"
    ]

n_hour = 30
n_day = n_hour * 24
n_year = n_day * 365

# Model parameters
parameters = Parameters(
    {
        "N_x": 1024,
        "N_y": 1024,
        "nu": 10.0,  # m^2/s
        "dt": 3600 / n_hour,  # s
        # Following from
        #     'Parameterization of ocean eddies: Potential vorticity mixing,
        #     energetics and Arnold's first stability theorem', David P.
        #     Marshall and Alistair J. Adcroft, Ocean Modelling 32(3-4),
        #     pp. 188--204, 2010, doi: 10.1016/j.ocemod.2010.02.001
        "L_x": 4000.0e3 / 2,  # m
        "L_y": 4000.0e3 / 2,  # m
        "beta": 2.0e-11,  # /m/s
        "r": 1.0e-7  # /s
    })

# Wind forcing parameters
rho_0 = 1.0e3  # kg/m^3
# Following from
#     'Parameterization of ocean eddies: Potential vorticity mixing, energetics
#     and Arnold's first stability theorem', David P. Marshall and Alistair J.
#     Adcroft, Ocean Modelling 32(3-4), pp. 188--204, 2010,
#     doi: 10.1016/j.ocemod.2010.02.001
D = 0.5e3  # m
tau_0 = 0.1  # N/m^2


def Q(grid):
    """Wind forcing term in the barotropic vorticity equation. Follows from
    the equation at the start of section 4.2 in

        - 'Parameterization of ocean eddies: Potential vorticity mixing,
          energetics and Arnold's first stability theorem', David P. Marshall
          and Alistair J. Adcroft, Ocean Modelling 32(3-4), pp. 188--204, 2010,
          doi: 10.1016/j.ocemod.2010.02.001

    Parameters
    ----------

    :class:`.Grid`
        The 2D Chebyshev grid.

    Returns
    -------

    :class:`jax.Array`
        The wind forcing term in the barotropic vorticity equation.
    """

    return ((tau_0 * jnp.pi / (D * rho_0 * grid.L_y))
            * jnp.sin(jnp.pi * grid.Y / grid.L_y))
