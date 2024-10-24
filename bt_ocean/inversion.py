"""Linear solvers.
"""

import jax
import jax.numpy as jnp
from jax.scipy.linalg import lu_factor, lu_solve
import numpy as np

from functools import cached_property

from .grid import Grid


__all__ = \
    [
        "KroneckerProductSolver",
        "ModifiedHelmholtzSolver",
        "PoissonSolver"
    ]


class KroneckerProductSolver:
    r"""Solver for a linear system with Kronecker product structure,

    .. math::

        ( I \otimes A + B \otimes I ) u = -b,

    using equation (3.9) of

        - Robert E. Lynch, John R. Rice and Donald H. Thomas, 'Direct solution
          of partial difference equations by tensor product methods',
          Numerische Mathematik 6, pp. 185--199, 1964,
          https://doi.org/10.1007/BF01386067

    Warnings
    --------

    Note the extra negative sign in the definition of the right-hand-side.

    Parameters
    ----------

    A : :class:`jax.Array`
        Defines the matrix :math:`A`.
    B : :class:`jax.Array`
        Defines the matrix :math:`B`.
    """

    def __init__(self, A, B):
        self._A = A
        self._B = B

        # Cached property initialization
        self._A_decomp
        self._B_decomp

    @staticmethod
    def _decomp(A):
        if isinstance(A, jax._src.core.Tracer):
            with jax.default_device(jax.devices("cpu")[0]):
                Lam, V = jnp.linalg.eig(A)
                Lam = Lam.real
                V = V.real
        else:
            Lam, V = np.linalg.eig(np.array(A, dtype=A.dtype))
            assert (Lam.imag == 0).all()
            assert (V.imag == 0).all()
            Lam = jnp.array(Lam.real, dtype=A.dtype)
            V = jnp.array(V.real, dtype=A.dtype)

        return Lam, V, lu_factor(V)

    @cached_property
    def _A_decomp(self):
        return self._decomp(self._A)

    @cached_property
    def _B_decomp(self):
        return self._decomp(self._B)

    @staticmethod
    @jax.jit
    def _solve(Lam_A, Q, Q_lu, Lam_B, P, P_lu, b):
        # Linear solve using equation (3.9) of
        #     Robert E. Lynch, John R. Rice and Donald H. Thomas, 'Direct
        #     solution of partial difference equations by tensor product
        #     methods', Numerische Mathematik 6, pp. 185--199, 1964,
        #     https://doi.org/10.1007/BF01386067

        u = lu_solve(Q_lu, -b.T).T
        u = lu_solve(P_lu, u)
        scale = (jnp.outer(jnp.ones_like(u, shape=(u.shape[0],)), Lam_A)
                 + jnp.outer(Lam_B, jnp.ones_like(u, shape=(u.shape[1],))))
        u = P @ (u / scale) @ Q.T
        return u

    def solve(self, b):
        """Solve the linear system.

        Computed using equation (3.9) of

            - Robert E. Lynch, John R. Rice and Donald H. Thomas, 'Direct
              solution of partial difference equations by tensor product
              methods', Numerische Mathematik 6, pp. 185--199, 1964,
              https://doi.org/10.1007/BF01386067

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

        return self._solve(*(self._A_decomp + self._B_decomp + (b,)))


class ModifiedHelmholtzSolver(KroneckerProductSolver):
    r"""Solver for the 2D modified Helmholtz equation,

    .. math::

        ( \alpha - \partial_{xx} - \partial_{yy} ) u = -b,

    subject to homogeneous Dirichlet boundary conditions, using a Chebyshev
    pseudospectral discretization.

    Parameters
    ----------

    grid : :class:`.Grid`
        The 2D Chebyshev grid.
    alpha : Real
        :math:`\alpha`.
    """

    def __init__(self, grid, alpha):
        A = 0.5 * alpha * jnp.eye(grid.N_y - 1, dtype=grid.fdtype) - grid.D_yy[1:-1, 1:-1]
        B = 0.5 * alpha * jnp.eye(grid.N_x - 1, dtype=grid.fdtype) - grid.D_xx[1:-1, 1:-1]

        super().__init__(A, B)
        self._grid = grid

    @property
    def grid(self) -> Grid:
        """The 2D Chebyshev grid."""

        return self._grid

    def solve(self, b):
        return jnp.zeros_like(b).at[1:-1, 1:-1].set(super().solve(b[1:-1, 1:-1]))


class PoissonSolver(ModifiedHelmholtzSolver):
    r"""Solver for the 2D Poisson equation,

    .. math::

        -( \partial_{xx} + \partial_{yy} ) u = -b,

    subject to homogeneous Dirichlet boundary conditions, using a Chebyshev
    pseudospectral discretization.

    Parameters
    ----------

    grid : :class:`.Grid`
        The 2D Chebyshev grid.
    """

    def __init__(self, grid):
        super().__init__(grid, alpha=0)
