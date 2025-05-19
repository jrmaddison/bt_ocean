"""Precision utilities.
"""

import jax.numpy as jnp

__all__ = \
    [
        "default_idtype",
        "default_fdtype"
    ]


def default_idtype():
    """Return the default integer scalar data type.

    Returns
    -------

    type
        The default integer scalar data type.
    """

    return jnp.array(0).dtype.type


def default_fdtype():
    """Return the default floating point scalar data type.

    Returns
    -------

    type
        The default floating point scalar data type.
    """

    return jnp.array(0.0).dtype.type
