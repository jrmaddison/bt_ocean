"""Linear solvers.
"""

import jax.numpy as jnp

from .fft import dst, idst
from .grid import Grid


__all__ = \
    [
        "ModifiedHelmholtzSolver",
        "PoissonSolver"
    ]


class ModifiedHelmholtzSolver:
    r"""Solver for the 2D modified Helmholtz equation,

    .. math::

        ( \beta ( \partial_{xx} + \partial_{yy} ) - \alpha  ) u = b,

    subject to homogeneous Dirichlet boundary conditions, using a finite
    difference discretization.

    Parameters
    ----------

    grid : :class:`.Grid`
        The 2D grid.
    alpha : Real
        :math:`\alpha`.
    beta : Real
        :math:`\beta`.
    """

    def __init__(self, grid, alpha, beta=1):
        self._grid = grid
        self._alpha = grid.fdtype(alpha)
        self._beta = grid.fdtype(beta)

    @property
    def grid(self) -> Grid:
        """The 2D grid."""

        return self._grid

    def solve(self, b):
        """Solve the linear system.

        Parameters
        ----------

        b : :class:`jax.Array`
            Defines :math:`b` appearing on the right-hand-side. An ndim 2
            array.

        Returns
        -------

        :class:`jax.Array`
            The solution :math:`u`. An ndim 2 array.
        """

        b_tilde = dst(dst(b, axis=0), axis=1)
        a_k = 2 * (jnp.cos(jnp.arange(1, self.grid.N_x, dtype=self.grid.fdtype) * jnp.pi / self.grid.N_x) - 1) * ((self.grid.N_x / (2 * self.grid.L_x)) ** 2)
        a_l = 2 * (jnp.cos(jnp.arange(1, self.grid.N_y, dtype=self.grid.fdtype) * jnp.pi / self.grid.N_y) - 1) * ((self.grid.N_y / (2 * self.grid.L_y)) ** 2)
        u_tilde = jnp.zeros_like(b).at[1:-1, 1:-1].set(
            -b_tilde[1:-1, 1:-1] / (self._alpha - self._beta * (jnp.outer(a_k, jnp.ones_like(a_l)) + jnp.outer(jnp.ones_like(a_k), a_l))))
        u = idst(idst(u_tilde, axis=0), axis=1)

        return u


class PoissonSolver(ModifiedHelmholtzSolver):
    r"""Solver for the 2D Poisson equation,

    .. math::

        ( \partial_{xx} + \partial_{yy} ) u = b,

    subject to homogeneous Dirichlet boundary conditions, using a finite
    difference discretization.

    Parameters
    ----------

    grid : :class:`.Grid`
        The 2D grid.
    """

    def __init__(self, grid):
        super().__init__(grid, alpha=0)
