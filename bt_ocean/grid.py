"""2D Chebyshev grids.
"""

import jax
import jax.numpy as jnp

from functools import cached_property
from numbers import Real

from .chebyshev import Chebyshev
from .precision import default_idtype, default_fdtype

__all__ = \
    [
        "Grid",
        "SpectralGridTransfer"
    ]


class Grid:
    r"""2D Chebyshev grid.

    Parameters
    ----------

    L_x : Real
        Defines the :math:`x`-dimension extents, :math:`x \in [ -L_x, L_x ]`.
    L_y : Real
        Defines the :math:`y`-dimension extents, :math:`y \in [ -L_y, L_y ]`.
    N_x : Integral
        :math:`x`-dimension degree.
    N_y : Integral
        :math:`y`-dimension degree.
    idtype : type
        Integer scalar data type. Defaults to `jax.numpy.int64` if 64-bit is
        enabled, and `jax.numpy.int32` otherwise.
    fdtype : type
        Floating point scalar data type. Defaults to `jax.numpy.float64` if
        64-bit is enabled, and `jax.numpy.float32` otherwise.
    """

    def __init__(self, L_x, L_y, N_x, N_y, *, idtype=None, fdtype=None):
        if idtype is None:
            idtype = default_idtype()
        if fdtype is None:
            fdtype = default_fdtype()

        cheb_x = Chebyshev(N_x, idtype=idtype, fdtype=fdtype)
        if N_y == N_x:
            cheb_y = cheb_x
        else:
            cheb_y = Chebyshev(N_y, idtype=idtype, fdtype=fdtype)

        self._L_x = fdtype(L_x)
        self._L_y = fdtype(L_y)
        self._cheb_x = cheb_x
        self._cheb_y = cheb_y
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
    def cheb_x(self) -> Chebyshev:
        """Defines the :math:`x`-dimension Chebsyshev grid.
        """

        return self._cheb_x

    @property
    def cheb_y(self) -> Chebyshev:
        """Defines the :math:`y`-dimension Chebsyshev grid.
        """

        return self._cheb_y

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
        """:math:`x`-dimension Chebsyshev degree.
        """

        return self.cheb_x.N

    @property
    def N_y(self) -> int:
        """:math:`y`-dimension Chebsyshev degree.
        """

        return self.cheb_y.N

    @cached_property
    def x(self) -> jax.Array:
        """:math:`x`-coordinates.
        """

        return self.cheb_x.x * self.L_x

    @cached_property
    def y(self) -> jax.Array:
        """:math:`y`-coordinates.
        """

        return self.cheb_y.x * self.L_y

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

    def interpolate(self, u, x, y, *, extrapolate=False):
        """Evaluate on a grid.

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

        v = self.cheb_x.interpolate(u, x / self.L_x, axis=0, extrapolate=extrapolate)
        v = self.cheb_y.interpolate(v, y / self.L_y, axis=1, extrapolate=extrapolate)
        return v

    @cached_property
    def W(self) -> jax.Array:
        """Integration matrix diagonal.
        """

        return jnp.outer(self.cheb_x.w, self.cheb_y.w) * self.L_x * self.L_y

    @cached_property
    def D_x(self) -> jax.Array:
        """:math:`x` direction first derivative matrix,
        """

        return self.cheb_x.D / self.L_x

    @cached_property
    def D_y(self) -> jax.Array:
        """:math:`y` direction first derivative matrix,
        """

        return self.cheb_y.D / self.L_y

    @cached_property
    def D_xx(self) -> jax.Array:
        """:math:`x` direction second derivative matrix,
        """

        return self.D_x @ self.D_x

    @cached_property
    def D_yy(self) -> jax.Array:
        """:math:`y` direction second derivative matrix,
        """

        return self.D_y @ self.D_y


class SpectralGridTransfer:
    """Grid-to-grid transfer. Down-scaling is performed by Chebyshev spectral
    truncation.

    Parameters
    ----------

    grid_a : :class:`.Grid`
        The lower degree grid.
    grid_b : :class:`.Grid`
        The higher degree grid.
    """

    def __init__(self, grid_a, grid_b):
        if grid_b.L_x != grid_a.L_x or grid_b.L_y != grid_a.L_y:
            raise ValueError("Invalid grids")
        if grid_b.N_x < grid_a.N_x or grid_b.N_y < grid_a.N_y:
            raise ValueError("Invalid grids")
        self._grid_a = grid_a
        self._grid_b = grid_b

    @property
    def grid_a(self) -> Grid:
        """The lower degree grid."""

        return self._grid_a

    @property
    def grid_b(self) -> Grid:
        """The higher degree grid.
        """

        return self._grid_b

    def to_higher_degree(self, u):
        """Transfer grid point values from the lower degree grid to the higher
        degree grid.

        Parameters
        ----------

        u : :class:`jax.Array`
            Array of grid point values on the lower degree grid.

        Returns
        -------

        :class:`jax.Array`
            Array of grid point values on the higher degree grid.
        """

        # Move to Chebyshev spectral basis
        u_c = self.grid_a.cheb_y.to_cheb(self.grid_a.cheb_x.to_cheb(u, axis=0), axis=1)
        # Extend
        u_c_e = jnp.zeros_like(u_c, shape=(self.grid_b.N_x + 1, self.grid_b.N_y + 1))
        u_c_e = u_c_e.at[:self.grid_a.N_x + 1, :self.grid_a.N_y + 1].set(u_c)
        # Return from Chebyshev spectral basis
        return self.grid_b.cheb_y.from_cheb(self.grid_b.cheb_x.from_cheb(u_c_e, axis=0), axis=1)

    def from_higher_degree(self, u):
        """Transfer grid point values from the higher degree grid to the lower
        degree grid via Chebyshev spectral truncation.

        Parameters
        ----------

        u : :class:`jax.Array`
            Array of grid point values on the higher degree grid.

        Returns
        -------

        :class:`jax.Array`
            Array of grid point values on the lower degree grid.
        """

        # Move to Chebyshev spectral basis
        u_c = self.grid_b.cheb_y.to_cheb(self.grid_b.cheb_x.to_cheb(u, axis=0), axis=1)
        # Truncate
        u_c_t = u_c[:self.grid_a.N_x + 1, :self.grid_a.N_y + 1]
        # Return from Chebyshev spectral basis
        return self.grid_a.cheb_y.from_cheb(self.grid_a.cheb_x.from_cheb(u_c_t, axis=0), axis=1)
