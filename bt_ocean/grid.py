"""2D grids.
"""

import jax
import jax.numpy as jnp

from functools import cached_property, partial
from numbers import Real

from .precision import default_idtype, default_fdtype

__all__ = \
    [
        "Grid"
    ]


class Grid:
    r"""2D grid.

    Parameters
    ----------

    L_x : Real
        Defines the :math:`x`-dimension extents, :math:`x \in [ -L_x, L_x ]`.
    L_y : Real
        Defines the :math:`y`-dimension extents, :math:`y \in [ -L_y, L_y ]`.
    N_x : Integral
        Number of :math:`x`-dimension divisions.
    N_y : Integral
        Number of :math:`y`-dimension divisions.
    idtype : type
        Integer scalar data type. Defaults to :func:`.default_idtype()`.
    fdtype : type
        Floating point scalar data type. Defaults to :func:`.default_fdtype()`.
    """

    def __init__(self, L_x, L_y, N_x, N_y, *, idtype=None, fdtype=None):
        if idtype is None:
            idtype = default_idtype()
        if fdtype is None:
            fdtype = default_fdtype()

        self._L_x = fdtype(L_x)
        self._L_y = fdtype(L_y)
        self._N_x = idtype(N_x)
        self._N_y = idtype(N_y)
        self._idtype = idtype
        self._fdtype = fdtype

    @property
    def L_x(self) -> Real:
        r""" Defines the :math:`x`-dimension extents, :math:`x \in
        [ -L_x, L_x ]`.
        """

        return self._L_x

    @property
    def L_y(self) -> Real:
        r""" Defines the :math:`y`-dimension extents, :math:`y \in
        [ -L_y, L_y ]`.
        """

        return self._L_y

    @property
    def idtype(self) -> type:
        """Integer scalar data type.
        """

        return self._idtype

    @property
    def fdtype(self) -> type:
        """Floating point scalar data type.
        """

        return self._fdtype

    @property
    def N_x(self) -> int:
        """Number of :math:`x`-dimension divisions.
        """

        return self._N_x

    @property
    def N_y(self) -> int:
        """Number of :math:`y`-dimension divisions.
        """

        return self._N_y

    @cached_property
    def x(self) -> jax.Array:
        """:math:`x`-coordinates.
        """

        return jnp.linspace(-self.L_x, self.L_x, self._N_x + 1, dtype=self.fdtype)

    @cached_property
    def y(self) -> jax.Array:
        """:math:`y`-coordinates.
        """

        return jnp.linspace(-self.L_y, self.L_y, self.N_y + 1, dtype=self.fdtype)

    @cached_property
    def X(self) -> jax.Array:
        """Grid :math:`x`-coordinates.
        """

        return jnp.outer(
            self.x, jnp.ones(self.N_y + 1, dtype=self.fdtype))

    @cached_property
    def Y(self) -> jax.Array:
        """Grid :math:`y`-coordinates.
        """

        return jnp.outer(
            jnp.ones(self.N_x + 1, dtype=self.fdtype), self.y)

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def _interpolate(idtype, x_g, y_g, u, x, y):
        N_x, = x_g.shape
        N_x -= 1
        N_y, = y_g.shape
        N_y -= 1
        L_x = x_g[-1]
        L_y = y_g[-1]

        i = jnp.array((x + L_x) * N_x / (2 * L_x), dtype=idtype)
        i = jnp.maximum(i, idtype(0))
        i = jnp.minimum(i, idtype(N_x - 1))
        alpha = (x - x_g[i]) / (x_g[i + 1] - x_g[i])

        j = jnp.array((y + L_y) * N_y / (2 * L_y), dtype=idtype)
        j = jnp.maximum(j, idtype(0))
        j = jnp.minimum(j, idtype(N_y - 1))
        beta = (y - y_g[j]) / (y_g[j + 1] - y_g[j])

        return (jnp.outer(1 - alpha, 1 - beta) * u[i, :][:, j]
                + jnp.outer(alpha, 1 - beta) * u[i + 1, :][:, j]
                + jnp.outer(1 - alpha, beta) * u[i, :][:, j + 1]
                + jnp.outer(alpha, beta) * u[i + 1, :][:, j + 1])

    def interpolate(self, u, x, y, *, extrapolate=False):
        """Bilinearly interpolate onto a grid.

        Parameters
        ----------

        u : :class:`jax.Array`
            Array of grid point values.
        x : :class:`jax.Array`
            :math:`x`-coordinates.
        y : :class:`jax.Array`
            :math:`y`-coordinates.
        extrapolate : bool
            Whether to allow extrapolation.

        Returns
        -------

        :class:`jax.Array`
            Array of values on the grid.
        """

        if not extrapolate and ((x < -self.L_x).any() or (x > self.L_x).any()
                                or (y < -self.L_y).any() or (y > self.L_y).any()):
            raise ValueError("Out of bounds")

        return self._interpolate(self.idtype, self.x, self.y, u, x, y)

    @cached_property
    def dx(self) -> Real:
        """:math:`x`-dimension grid spacing.
        """

        return 2 * self.L_x / self.N_x

    @cached_property
    def dy(self) -> Real:
        """:math:`y`-dimension grid spacing.
        """

        return 2 * self.L_y / self.N_y

    @cached_property
    def W(self) -> jax.Array:
        """Integration matrix diagonal.
        """

        w_x = jnp.ones(self.N_x + 1, dtype=self.fdtype) * self.dx
        w_x = w_x.at[0].set(0.5 * self.dx)
        w_x = w_x.at[-1].set(0.5 * self.dx)

        w_y = jnp.ones(self.N_y + 1, dtype=self.fdtype) * self.dy
        w_y = w_y.at[0].set(0.5 * self.dy)
        w_y = w_y.at[-1].set(0.5 * self.dy)

        return jnp.outer(w_x, w_y)

    def D_x(self, u, *, boundary=True):
        """Compute an :math:`x`-direction first derivative.

        Parameters
        ----------

        u : :class:`jax.Array`
            Field to differentiate.
        boundary : bool
            Whether to compute the derivative on the left and right boundaries.

        Returns
        -------

        :class:`jax.Array`
            The derivative.
        """

        D = jnp.zeros_like(u).at[1:-1, :].set(
            (u[2:, :] - u[:-2, :]) / (2 * self.dx))
        if boundary:
            D = D.at[0, :].set((-2.5 * u[1, :] + 4 * u[2, :] - 1.5 * u[3, :]) / self.dx)
            D = D.at[-1, :].set(-(-2.5 * u[-2, :] + 4 * u[-3, :] - 1.5 * u[-4, :]) / self.dx)
        return D

    def D_y(self, u, boundary=True):
        """Compute an :math:`y`-direction first derivative.

        Parameters
        ----------

        u : :class:`jax.Array`
            Field to differentiate.
        boundary : bool
            Whether to compute the derivative on the top and bottom boundaries.

        Returns
        -------

        :class:`jax.Array`
            The derivative.
        """

        D = jnp.zeros_like(u).at[:, 1:-1].set(
            (u[:, 2:] - u[:, :-2]) / (2 * self.dy))
        if boundary:
            D = D.at[:, 0].set((-2.5 * u[:, 1] + 4 * u[:, 2] - 1.5 * u[:, 3]) / self.dy)
            D = D.at[:, -1].set(-(-2.5 * u[:, -2] + 4 * u[:, -3] - 1.5 * u[:, -4]) / self.dy)
        return D

    def D_xx(self, u, boundary=True):
        """Compute an :math:`x`-direction second derivative.

        Parameters
        ----------

        u : :class:`jax.Array`
            Field to differentiate.
        boundary : bool
            Whether to compute the derivative on the left and right boundaries.

        Returns
        -------

        :class:`jax.Array`
            The derivative.
        """

        D = jnp.zeros_like(u).at[1:-1, :].set(
            (u[2:, :] - 2 * u[1:-1, :] + u[:-2, :]) / (self.dx ** 2))
        if boundary:
            D = D.at[0, :].set((3 * u[1, :] - 8 * u[2, :] + 7 * u[3, :] - 2 * u[4, :]) / (self.dx ** 2))
            D = D.at[-1, :].set((3 * u[-2, :] - 8 * u[-3, :] + 7 * u[-4, :] - 2 * u[-5, :]) / (self.dx ** 2))
        return D

    def D_yy(self, u, boundary=True):
        """Compute a :math:`y`-direction second derivative.

        Parameters
        ----------

        u : :class:`jax.Array`
            Field to differentiate.
        boundary : bool
            Whether to compute the derivative on the top and bottom boundaries.

        Returns
        -------

        :class:`jax.Array`
            The derivative.
        """

        D = jnp.zeros_like(u).at[:, 1:-1].set(
            (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / (self.dy ** 2))
        if boundary:
            D = D.at[:, 0].set((3 * u[:, 1] - 8 * u[:, 2] + 7 * u[:, 3] - 2 * u[:, 4]) / (self.dy ** 2))
            D = D.at[:, -1].set((3 * u[:, -2] - 8 * u[:, -3] + 7 * u[:, -4] - 2 * u[:, -5]) / (self.dy ** 2))
        return D

    def integrate(self, u):
        """
        Compute the integral of a field.

        Parameters
        ----------

        u : :class:`jax.Array`
            Field to integrate.

        Returns
        -------

        :class:`jax.Array`
            The integral.
        """

        return jnp.tensordot(u, self.W)

    @staticmethod
    @jax.jit
    def _J(dx, dy, q, psi):
        N_x, N_y = q.shape
        N_x -= 1
        N_y -= 1

        # Equations (36)--(38), (40), and (44) in
        #     Akio Arakawa, 'Computational design for long-term numerical
        #     integration of the equations of fluid motion: two-dimensional
        #     incompressible flow. Part I', Journal of Computational
        #     Physics 1(1), 119--143, 1966
        return jnp.zeros_like(q).at[1:-1, 1:-1].set(
            (1 / (12 * dx * dy)) * (
                + (q[2:, 1:-1] - q[:-2, 1:-1]) * (psi[1:-1, 2:] - psi[1:-1, :-2])
                - (q[1:-1, 2:] - q[1:-1, :-2]) * (psi[2:, 1:-1] - psi[:-2, 1:-1])
                + q[2:, 1:-1] * (psi[2:, 2:] - psi[2:, :-2])
                - q[:-2, 1:-1] * (psi[:-2, 2:] - psi[:-2, :-2])
                - q[1:-1, 2:] * (psi[2:, 2:] - psi[:-2, 2:])
                + q[1:-1, :-2] * (psi[2:, :-2] - psi[:-2, :-2])
                + q[2:, 2:] * (psi[1:-1, 2:] - psi[2:, 1:-1])
                - q[:-2, :-2] * (psi[:-2, 1:-1] - psi[1:-1, :-2])
                - q[:-2, 2:] * (psi[1:-1, 2:] - psi[:-2, 1:-1])
                + q[2:, :-2] * (psi[2:, 1:-1] - psi[1:-1, :-2])))

    def J(self, q, psi):
        """Arakawa Jacobian, using equations (36)--(38), (40), and (44) in

            - Akio Arakawa, 'Computational design for long-term numerical
              integration of the equations of fluid motion: two-dimensional
              incompressible flow. Part I', Journal of Computational Physics
              1(1), 119--143, 1966
        """

        return self._J(self.dx, self.dy, q, psi)
