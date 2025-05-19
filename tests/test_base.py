import jax.numpy as jnp

from bt_ocean.precision import default_fdtype


__all__ = \
    [
        "eps"
    ]


def eps():
    return jnp.finfo(default_fdtype()).eps
