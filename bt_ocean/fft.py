"""FFT utilities.
"""

import jax
import jax.numpy as jnp

from functools import partial

__all__ = \
    [
        "dst",
        "idst"
    ]


@partial(jax.jit, static_argnames="axis")
def dst(u, *, axis=-1):
    """Type-I discrete sine transform.

    Parameters
    ----------

    u : :class:`jax.Array`
        Field to transform.
    axis : Integral
        Axis over which to perform the transform.

    Returns
    -------

    :class:`jax.Array`
        The result of the transform. Has the same shape as `u`.
    """

    if axis == -1:
        axis = len(u.shape) - 1
    if axis != len(u.shape) - 1:
        p = tuple(range(axis)) + tuple(range(axis + 1, len(u.shape))) + (axis,)
        p_inv = tuple(range(axis)) + (len(u.shape) - 1,) + tuple(range(axis, len(u.shape) - 1))
        u = jnp.transpose(u, p)

    N = u.shape[-1] - 1
    u_e = jnp.zeros_like(u, shape=u.shape[:-1] + (2 * N,))
    u_e = u_e.at[..., :N + 1].set(-u)
    u_e = u_e.at[..., N + 1:].set(jnp.flip(u[..., 1:-1], axis=-1))
    u_s = jnp.fft.rfft(u_e, axis=-1).imag
    u_s = u_s.at[..., 0].set(0)
    u_s = u_s.at[..., -1].set(0)
    u_s = u_s / N

    if axis != len(u.shape) - 1:
        u_s = jnp.transpose(u_s, p_inv)
    return u_s


@partial(jax.jit, static_argnames="axis")
def idst(u_s, *, axis=-1):
    """Type-I discrete sine transform inverse. Inverse of :func:`.dst`.

    Parameters
    ----------

    u : :class:`jax.Array`
        Field to transform.
    axis : Integral
        Axis over which to perform the transform.

    Returns
    -------

    :class:`jax.Array`
        The result of the transform. Has the same shape as `u`.
    """

    return dst(0.5 * (u_s.shape[axis] - 1) * u_s, axis=axis)
