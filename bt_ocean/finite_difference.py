"""Finite difference utilities.

Finite difference coefficients are found as in equation (1.19) in

    - Randall J. LeVeque, 'Finite difference methods for ordinary and partial
      differential equations', Society for Industrial and Applied Mathematics,
      2007
"""

from functools import lru_cache, partial
from numbers import Complex, Rational, Real

import jax
import jax.numpy as jnp
import sympy as sp

__all__ = \
    [
        "difference_coefficients",
        "order_reversed",
        "diff_bounded",
        "diff_periodic"
    ]


def difference_coefficients(beta, order):
    """Compute 1D finite difference coefficients of maximal order of accuracy.

     Finite difference coefficients are found as in equation (1.19) in

        - Randall J. LeVeque, 'Finite difference methods for ordinary and
          partial differential equations', Society for Industrial and Applied
          Mathematics, 2007

    Parameters
    ----------

    beta : Sequence
        Grid location displacements.
    order : Integral
        Derivative order.

    Returns
    -------

    tuple[:class:`sympy.core.expr.Expr`, ...]
        Finite difference coefficients.
    """

    def displacement_cast(v):
        if isinstance(v, Rational):
            return sp.Rational(v)
        else:
            return v

    return _difference_coefficients(tuple(map(displacement_cast, beta)), order)


@lru_cache(maxsize=32)
def _difference_coefficients(beta, order):
    N = len(beta)
    if order < 0 or order >= N:
        raise ValueError("Invalid order")

    def is_real(v):
        if isinstance(v, Real):
            return True
        elif isinstance(v, Complex):
            return False
        elif isinstance(v, sp.core.expr.Expr):
            return v.is_real
        else:
            return None

    assumptions = {}
    if all(map(bool, map(is_real, beta))):
        assumptions["real"] = True
    a = tuple(sp.Symbol("_bt_ocean__finite_difference_{" + f"{i}" + "}", **assumptions)
              for i in range(N))

    # Equation (1.19) in
    #     Randall J. LeVeque, 'Finite difference methods for ordinary and
    #     partial differential equations', Society for Industrial and Applied
    #     Mathematics, 2007
    eqs = [sum((a[i] * ((beta[i] ** j) / sp.factorial(j))
                for i in range(N)), start=sp.Integer(0))
           for j in range(N)]
    eqs[order] -= sp.Integer(1)

    soln, = sp.linsolve(eqs, a)
    return soln


def order_reversed(alpha, beta):
    """Defines a reversed grid point ordering. Can be used as the
    `interior_order` or `boundary_order` argument for :func:`.diff_bounded`.

    Parameters
    ----------

    alpha : Sequence
        Coefficients.
    beta : Sequence
        Displacements

    Returns
    -------

    Sequence
        Reordered coefficients.
    Sequence
        Reordered displacements.
    """

    return tuple(reversed(alpha)), tuple(reversed(beta))


@partial(jax.jit, static_argnames={"order", "N", "axis", "i0", "i1", "boundary_expansion", "interior_order", "boundary_order"})
def diff_bounded(u, dx, order, N, *, axis=-1, i0=None, i1=None, boundary_expansion=None,
                 interior_order=None, boundary_order=None):
    """Compute a centred finite difference approximation for a derivative for
    data stored on a uniform grid. Result is defined on the same grid as the
    input (i.e. without staggering). Transitions to one-sided differencing as
    the end-points are approached.

    Parameters
    ----------

    u : :class:`jax.Array`
        Field to difference.
    dx : Real
        Grid spacing.
    order : Integral
        Derivative order.
    N : Integral
        Number of grid points in the difference approximation. Centered
        differencing uses an additional right-sided point if `N` is even.
    axis : Integral
        Axis.
    i0 : Integral
        Index lower limit. Values with index less than the index defined by
        `i0` are set to zero.
    i1 : Integral
        Index upper limit. Values with index greater than or equal to the index
        defined by `i1` are set to zero.
    boundary_expansion : bool
        Whether to use one additional grid point for one-sided differencing
        near the boundary. Defaults to `True` if `order` is even and `False`
        otherwise.
    interior_order : callable
        Used to define an ordering for interior grid points. See
        :func:`.ordering_reversed` for an example. Note that a positive
        displacement corresponds to increasing index.
    boundary_order : callable
        Used to define an ordering for boundary grid points. See
        :func:`.ordering_reversed` for an example. Note that a positive
        displacement corresponds to a domain inward direction.

    Returns
    -------

    :class:`jax.Array`
        Finite difference approximation.
    """

    u = jnp.moveaxis(u, axis, -1)

    if boundary_expansion is None:
        boundary_expansion = (order % 2) == 0
    if u.shape[-1] < N + int(bool(boundary_expansion)):
        raise ValueError("Insufficient points")

    i0_b, i1_b = i0, i1
    del i0, i1
    if i0_b is None:
        i0_b = 0
    elif i0_b < 0:
        i0_b = u.shape[-1] + i0_b
    if i1_b is None:
        i1_b = u.shape[-1]
    elif i1_b < 0:
        i1_b = u.shape[-1] + i1_b

    v = jnp.zeros_like(u)
    dtype = u.dtype.type
    i0 = -(N // 2)
    i1 = i0 + N
    assert i1 > 0  # Insufficient points
    parity = (-1) ** order

    for i in range(max(0, min(i0_b, u.shape[-1] - i1_b)), max(-i0, i1 - 1)):
        beta = tuple(range(-i, -i + N + int(bool(boundary_expansion))))
        alpha = tuple(map(dtype, difference_coefficients(beta, order)))
        if boundary_order is not None:
            alpha, beta = boundary_order(alpha, beta)
        if i < -i0 and i >= i0_b:
            # Left end points
            assert len(alpha) == len(beta)
            for alpha_j, beta_j in zip(alpha, beta):
                v = v.at[..., i].add(alpha_j * u[..., i + beta_j])
        if i < i1 - 1 and u.shape[-1] - 1 - i < i1_b:
            # Right end points
            assert len(alpha) == len(beta)
            for alpha_j, beta_j in zip(alpha, beta):
                v = v.at[..., u.shape[-1] - 1 - i].add(
                    parity * alpha_j * u[..., u.shape[-1] - 1 - i - beta_j])

    # Center points
    beta = tuple(range(i0, i1))
    alpha = tuple(map(dtype, difference_coefficients(beta, order)))
    if interior_order is not None:
        alpha, beta = interior_order(alpha, beta)
    i0_c = max(-i0, i0_b)
    i1_c = min(u.shape[-1] - i1 + 1, i1_b)
    assert len(alpha) == len(beta)
    for alpha_i, beta_i in zip(alpha, beta):
        v = v.at[..., i0_c:i1_c].add(
            alpha_i * u[..., i0_c + beta_i:i1_c + beta_i])

    v = jnp.moveaxis(v, -1, axis)
    return v / (dx ** order)


@partial(jax.jit, static_argnames={"order", "N", "axis", "interior_order"})
def diff_periodic(u, dx, order, N, *, axis=-1, interior_order=None):
    """Compute a centred finite difference approximation for a derivative for
    data stored on a uniform grid. Result is defined on the same grid as the
    input (i.e. without staggering). Applies periodic boundary conditions.

    Arguments and return value are as for :func:`.diff_bounded`.
    """

    u = jnp.moveaxis(u, axis, -1)

    if u.shape[-1] < N:
        raise ValueError("Insufficient points")

    i0 = -(N // 2)
    i1 = i0 + N
    assert i1 > 0  # Insufficient points

    # Periodic extension
    u_e = jnp.zeros_like(u, shape=u.shape[:-1] + (u.shape[-1] + N,))
    u_e = u_e.at[..., -i0:-i1].set(u)
    u_e = u_e.at[..., :-i0].set(u[..., i0:])
    u_e = u_e.at[..., -i1:].set(u[..., :i1])

    v = diff_bounded(u_e, dx, order, N, axis=-1, i0=-i0, i1=-i1, interior_order=interior_order)[..., -i0:-i1]

    v = jnp.moveaxis(v, -1, axis)
    return v
