"""Chebyshev pseudospectral utilities.
"""

import jax
import jax.numpy as jnp

from enum import Enum
from functools import cached_property, partial

from .fft import dchebt, idchebt
from .precision import default_idtype, default_fdtype

__all__ = \
    [
        "InterpolationMethod",
        "Chebyshev"
    ]


class InterpolationMethod(Enum):
    """Method to use for interpolation. See :meth:`.Chebyshev.interpolate`.
    """

    BARYCENTRIC = 0
    CLENSHAW = 1


class Chebyshev:
    """Chebyshev pseudospectral utility class.

    Parameters
    ----------

    N : Integral
        Degree. Number of grid points minus one.
    idtype : type
        Integer scalar data type. Defaults to `jax.numpy.int64` if 64-bit is
        enabled, and `jax.numpy.int32` otherwise.
    fdtype : type
        Floating point scalar data type. Defaults to `jax.numpy.float64` if
        64-bit is enabled, and `jax.numpy.float32` otherwise.
    """

    def __init__(self, N, *, idtype=None, fdtype=None):
        self._N = int(N)
        self._idtype = default_idtype() if idtype is None else idtype
        self._fdtype = default_fdtype() if fdtype is None else fdtype

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
    def N(self) -> int:
        """Degree. Number of grid points minus one.
        """

        return self._N

    @cached_property
    def x(self) -> jax.Array:
        """Chebyshev grid point locations.

        Computed using equation (6.1) in

            - Lloyd N. Trefethen, 'Spectral methods in MATLAB', Society for
              Industrial and Applied Mathematics, 2000,
              https://doi.org/10.1137/1.9780898719598

        with an extra negative sign so that the values are in increasing order.
        """

        # Equation (6.1) in
        #     Lloyd N. Trefethen, 'Spectral methods in MATLAB', Society for
        #     Industrial and Applied Mathematics, 2000,
        #     https://doi.org/10.1137/1.9780898719598
        # with an extra negative sign
        j = jnp.arange(self.N + 1, dtype=self.idtype)
        x = -jnp.array(jnp.cos(j * jnp.pi / self.N), dtype=self.fdtype)

        # Symmetrize
        x = (x - jnp.flip(x)) / 2
        x = x.at[:(self.N + 1) // 2].set(-jnp.flip(x[-((self.N + 1) // 2):]))
        # Exact values for end points and mid-point
        x = x.at[0].set(-1)
        if self.N % 2 == 0:
            x = x.at[self.N // 2].set(0)
        x = x.at[-1].set(1)

        return x

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def _w_c(fdtype, n):
        # From
        #
        #     import sympy as sp
        #
        #     x_s = sp.Symbol("x_s", real=True)
        #     n_s = sp.Symbol("n_s", integer=True, negative=False)
        #     T_n = sp.cos(n_s * sp.acos(x_s))
        #     T_int_n = sp.simplify(sp.integrate(T_n, (x_s, -1, 1)))
        #     sp.pretty_print(T_int_n)
        #
        # SymPy 1.9
        return jnp.where(n != 1, jnp.array(-(1 - (-1) ** (n + 1)) / (n ** 2 - 1), dtype=fdtype), fdtype(0))

    @cached_property
    def w(self) -> jax.Array:
        """Clenshaw-Curtis quadrature weights.
        """

        w_c = self._w_c(self.fdtype, jnp.arange(self.N + 1, dtype=self.idtype))

        # The adjoint of the grid-to-spectral transform
        w_c = w_c.at[0].set(w_c[0] / 2)
        w_c = w_c.at[-1].set(w_c[-1] / 2)
        w_c = w_c / self.N
        w = self.from_cheb(w_c)
        w = w.at[1:-1].set(2 * w[1:-1])

        return w

    def to_cheb(self, u, *, axis=-1):
        """Transform the given array of grid point values to an array of
        expansions in the Chebyshev spectral basis. Uses :func:`.dchebt` to
        perform the transform.

        Parameters
        ----------

        u : :class:`jax.Array`
            Array of grid point values.
        axis : Integral
            Axis over which to perform the transform.

        Returns
        -------

        :class:`jax.Array`
            Array of expansions in the Chebyshev spectral basis.
        """

        if u.shape[axis] != self.N + 1:
            raise ValueError("Invalid shape")
        return dchebt(u, axis=axis)

    def from_cheb(self, u_c, *, axis=-1):
        """Transform the given array of expansions in the Chebyshev spectral
        basis to an array of grid point values. Uses :func:`.idchebt` to
        perform the transform.

        Inverse of :meth:`to_cheb`.

        Parameters
        ----------

        u_c : :class:`jax.Array`
            Array of expansions in the Chebyshev spectral basis.
        axis : Integral
            Axis over which to perform the transform.

        Returns
        -------

        :class:`jax.Array`
            Array of grid point values.
        """

        if u_c.shape[axis] != self.N + 1:
            raise ValueError("Invalid shape")
        return idchebt(u_c, axis=axis)

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1), static_argnames="axis")
    def _interpolate(idtype, fdtype, u, x, x_c, *, axis=-1):
        if axis == -1:
            axis = len(u.shape) - 1
        if axis != len(u.shape) - 1:
            p = tuple(range(axis)) + tuple(range(axis + 1, len(u.shape))) + (axis,)
            p_inv = tuple(range(axis)) + (len(u.shape) - 1,) + tuple(range(axis, len(u.shape) - 1))
            u = jnp.transpose(u, p)

        N = u.shape[-1] - 1

        # Equation (5.4) in
        #     Jean-Paul Berrut and Lloyd N. Trefethen, 'Barycentric Lagrange
        #     interpolation', SIAM Review 46(3), pp. 501--517, 2004,
        #     https://doi.org/10.1137/S0036144502417715
        # Note that coordinate order reversal might change the sign here, but
        # this does not change the result due to cancelling factors
        w = jnp.array((-1) ** jnp.arange(N + 1, dtype=idtype), dtype=fdtype)
        w = w.at[0].set(0.5 * w[0])
        w = w.at[-1].set(0.5 * w[-1])

        # Equation (4.2) in
        #     Jean-Paul Berrut and Lloyd N. Trefethen, 'Barycentric Lagrange
        #     interpolation', SIAM Review 46(3), pp. 501--517, 2004,
        #     https://doi.org/10.1137/S0036144502417715
        # When an interpolation point lies within eps of a grid point we
        # arbitrarily replace the divide-by-zero with divide-by-one. The
        # erroneous result will be discarded below.
        X = jnp.array(jnp.tensordot(x, jnp.ones_like(x, shape=x_c.shape), axes=0), dtype=fdtype)
        X_exact = jnp.array(abs(X - x_c) < jnp.finfo(fdtype).eps, dtype=idtype)
        a = w / jnp.where(X_exact == 1, jnp.ones_like(X), X - x_c)
        v = jnp.tensordot(u, a, axes=((-1,), (-1,))) / a.sum(axis=-1)

        # Now handle the case where an interpolation point lies within eps of a
        # grid point
        v = jnp.where(jnp.tensordot(jnp.ones(u.shape, dtype=idtype), X_exact, axes=((-1,), (-1,))) == 1,
                      jnp.tensordot(u, X_exact, axes=((-1,), (-1,))),
                      v)

        if axis != len(u.shape) - 1:
            v = jnp.transpose(v, p_inv)
        return v

    def interpolate(self, u, x, *, axis=-1, interpolation_method=InterpolationMethod.BARYCENTRIC, extrapolate=False):
        """Evaluate at the given locations, given an array of grid point
        values.

        The `interpolation_method` argument chooses between two interpolation
        methods:

            - `InterpolationMethod.BARYCENTRIC`: Barycentric interpolation as
              described in Jean-Paul Berrut and Lloyd N. Trefethen,
              'Barycentric Lagrange interpolation', SIAM Review 46(3),
              pp. 501--517, 2004, https://doi.org/10.1137/S0036144502417715,
              in particular using their equations (4.2) and (5.4).
            - `InterpolationMethod.CLENSHAW`: Interpolation performed by
              first transforming to an expansion in the Chebyshev basis using
              :meth:`.to_cheb`, and then using using the Clenshaw algorithm,
              using equations (2) and (3) in C. W. Clenshaw, 'A note on the
              summation of Chebyshev series', Mathematics of Computation 9,
              118--120, 1955.

        Parameters
        ----------

        u : :class:`jax.Array`
            Array of grid point values.
        x : :class:`jax.Array`
            Array of locations.
        axis : Integral
            Axis over which to perform the evaluation.
        interpolation_method
            The interpolation method.
        extrapolate : bool
            Whether to allow extrapolation.

        Returns
        -------

        :class:`jax.Array`
            Array of values at the given locations.
        """

        if u.shape[axis] != self.N + 1:
            raise ValueError("Invalid shape")
        if not extrapolate and (abs(x) > 1).any():
            raise ValueError("Out of bounds")
        if interpolation_method == InterpolationMethod.BARYCENTRIC:
            return self._interpolate(self.idtype, self.fdtype, u, x, self.x, axis=axis)
        elif interpolation_method == InterpolationMethod.CLENSHAW:
            return self.interpolate_cheb(self.to_cheb(u, axis=axis), x, axis=axis, extrapolate=extrapolate)
        else:
            raise ValueError(f"Unrecognized interpolation method: {interpolation_method}")

    @staticmethod
    @partial(jax.jit, static_argnames="axis")
    def _interpolate_cheb(u_c, x, *, axis=-1):
        if axis == -1:
            axis = len(u_c.shape) - 1
        if axis != len(u_c.shape) - 1:
            p = tuple(range(axis)) + tuple(range(axis + 1, len(u_c.shape))) + (axis,)
            p_inv = tuple(range(axis)) + (len(u_c.shape) - 1,) + tuple(range(axis, len(u_c.shape) - 1))
            u_c = jnp.transpose(u_c, p)

        N = u_c.shape[-1] - 1
        x0 = jnp.ones_like(x)

        def step(i, data):
            i = N - i
            b0, b1 = data
            # Equation (2) in
            #     C. W. Clenshaw, 'A note on the summation of Chebyshev
            #     series', Mathematics of Computation 9, 118--120, 1955
            b0, b1 = jnp.tensordot(u_c[..., i], x0, axes=0) + 2 * b0 * x - b1, b0
            return (b0, b1)

        b0 = b1 = jnp.zeros_like(u_c, shape=u_c.shape[:-1] + (x.shape[0],))
        b0, b1 = jax.lax.fori_loop(0, N + 1, step, (b0, b1), unroll=32)
        # Equation (3) in
        #     C. W. Clenshaw, 'A note on the summation of Chebyshev series',
        #     Mathematics of Computation 9, 118--120, 1955
        v = b0 - b1 * x

        if axis != len(u_c.shape) - 1:
            v = jnp.transpose(v, p_inv)
        return v

    def interpolate_cheb(self, u_c, x, *, axis=-1, extrapolate=False):
        """Evaluate at the given locations, given an array of expansions in the
        Chebyshev spectral basis.

        Computed using the Clenshaw algorithm, using equations (2) and (3) in

            - C. W. Clenshaw, 'A note on the summation of Chebyshev series',
              Mathematics of Computation 9, 118--120, 1955

        Parameters
        ----------

        u_c : :class:`jax.Array`
            Array of expansions in the Chebyshev spectral basis.
        x : :class:`jax.Array`
            Array of locations.
        axis : Integral
            Axis over which to perform the evaluation.
        extrapolate : bool
            Whether to allow extrapolation.

        Returns
        -------

        :class:`jax.Array`
            Array of values at the given locations.
        """

        if u_c.shape[axis] != self.N + 1:
            raise ValueError("Invalid shape")
        if not extrapolate and (abs(x) > 1).any():
            raise ValueError("Out of bounds")
        return self._interpolate_cheb(u_c, x, axis=axis)

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def _D(idtype, fdtype, x):
        N = x.shape[0] - 1
        # Equation (6.5) in
        #     Lloyd N. Trefethen, 'Spectral methods in MATLAB', Society for
        #     Industrial and Applied Mathematics, 2000,
        #     https://doi.org/10.1137/1.9780898719598
        # Coordinate order reversal does not change the expression for the
        # differentiation matrix
        c = jnp.ones_like(x)
        c = c.at[0].set(2)
        c = c.at[-1].set(2)
        i = jnp.arange(N + 1, dtype=idtype)
        X = jnp.outer(x, jnp.ones_like(x))
        C = jnp.outer(c, jnp.ones_like(c))
        I = jnp.outer(i, jnp.ones_like(i))
        fact = (-1) ** (I + I.T) * C / C.T
        D = jnp.where(I - I.T != 0, fact / jnp.where(I - I.T != 0, X - X.T, fdtype(1)), fdtype(0))
        # Equation (6.6) in
        #     Lloyd N. Trefethen, 'Spectral methods in MATLAB', Society for
        #     Industrial and Applied Mathematics, 2000,
        #     https://doi.org/10.1137/1.9780898719598
        D = D - jnp.diag(D.sum(1))
        return D

    @cached_property
    def D(self) -> jax.Array:
        """Chebyshev pseudospectral differentiation matrix.

        Computed using equations (6.5) and (6.6) in

            - Lloyd N. Trefethen, 'Spectral methods in MATLAB', Society for
              Industrial and Applied Mathematics, 2000,
              https://doi.org/10.1137/1.9780898719598
        """

        return self._D(self.idtype, self.fdtype, self.x)
