"""Finite difference utilities.
"""

from functools import partial
from numbers import Real

import jax
import jax.numpy as jnp
import sympy as sp

__all__ = \
    [
        "diff"
    ]


def difference_coefficients(beta, order):
    """Compute 1D finite difference coefficients of maximal order of accuracy.

    Parameters
    ----------

    beta : Sequence[Real, ...]
        Grid location displacements.
    order : Integral
        Derivative order.

    Returns
    -------

    Sequence[:class:`sympy.Rational`, ...]
        Finite difference coefficients.
    """

    beta = tuple(map(sp.Rational, beta))
    if not all(isinstance(beta_i, Real) for beta_i in beta):
        raise ValueError("Invalid type")
    N = len(beta)
    if order >= N:
        raise ValueError("Invalid order")

    a = tuple(sp.Symbol("{a_" + f"{i}" + "}", real=True)
              for i in range(N))
    eqs = [sum((a[i] * sp.Rational(beta[i] ** j, sp.factorial(j))
                for i in range(N)), start=sp.Integer(0))
           for j in range(N)]
    eqs[order] -= sp.Integer(1)

    soln, = sp.linsolve(eqs, a)
    return tuple(map(sp.Rational, soln))


@partial(jax.jit, static_argnames={"order", "N", "axis", "i0", "i1", "boundary_expansion"})
def diff(u, dx, order, N, *, axis=-1, i0=None, i1=None, boundary_expansion=None):
    """Compute a centred finite difference approximation to a derivative for
    data stored on a uniform grid. Transitions to one-sided differencing as the
    end-points are approached. Selects an additional right-sided point if
    `N` is even.

    Parameters
    ----------

    u : :class:`jax.Array`
        Field to difference.
    dx : Real
        Grid spacing.
    order : Integral
        Derivative order.
    N : Integral
        Number of grid points in the difference approximation.
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

    Returns
    -------

    :class:`jax.Array`
        Finite difference approximation.
    """

    if axis < 0:
        axis = len(u.shape) + axis
    if axis < 0 or axis >= len(u.shape):
        raise ValueError("Invalid axis")
    if boundary_expansion is None:
        boundary_expansion = (order % 2) == 0
    if u.shape[axis] < N + int(bool(boundary_expansion)):
        raise ValueError("Insufficient points")

    i0_b, i1_b = i0, i1
    del i0, i1
    if i0_b is None:
        i0_b = 0
    elif i0_b < 0:
        i0_b = u.shape[axis] + i0_b
    if i1_b is None:
        i1_b = u.shape[axis]
    elif i1_b < 0:
        i1_b = u.shape[axis] + i1_b

    u = jnp.moveaxis(u, axis, -1)

    v = jnp.zeros_like(u)
    dtype = u.dtype.type
    i0 = -(N // 2)
    i1 = i0 + N
    parity = (-1) ** order

    for i in range(max(-i0, i1 - 1)):
        beta = tuple(range(-i, -i + N + int(bool(boundary_expansion))))
        alpha = tuple(map(dtype, difference_coefficients(beta, order)))
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

    # Center
    beta = tuple(range(i0, i1))
    alpha = tuple(map(dtype, difference_coefficients(beta, order)))
    i0_c = max(-i0, i0_b)
    i1_c = min(u.shape[-1] - i1 + 1, i1_b)
    assert len(alpha) == len(beta)
    for alpha_i, beta_i in zip(alpha, beta):
        v = v.at[..., i0_c:i1_c].add(
            alpha_i * u[..., i0_c + beta_i:i1_c + beta_i])

    v = jnp.moveaxis(v, -1, axis)
    return v / (dx ** order)
