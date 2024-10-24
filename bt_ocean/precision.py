"""Precision utilities.
"""

from contextlib import contextmanager

import jax
import jax.numpy as jnp

__all__ = \
    [
        "x64_enabled",

        "default_idtype",
        "default_fdtype"
    ]


@contextmanager
def x64_enabled():
    """Context manager for temporarily enabling the 'jax_enable_x64' JAX
    configuration option.
    """

    x64_enabled = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", x64_enabled)


def default_idtype():
    """Return the current default integer scalar data type.

    Returns
    -------

    type
        The current default integer scalar data type.
    """

    return jnp.array(0).dtype.type


def default_fdtype():
    """Return the current default floating point scalar data type.

    Returns
    -------

    type
        The current default floating point scalar data type.
    """

    return jnp.array(0.0).dtype.type
