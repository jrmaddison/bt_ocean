"""Precision utilities.
"""

from contextlib import contextmanager

import jax
import jax.numpy as jnp
import keras

__all__ = \
    [
        "x64_disabled",
        "x64_enabled",

        "default_idtype",
        "default_fdtype"
    ]


@contextmanager
def x64_disabled():
    """Context manager for temporarily disabling the `'jax_enable_x64'` JAX
    configuration option, and for temporarily setting the Keras default float
    type to single precision.
    """

    x64_enabled = jax.config.x64_enabled
    floatx = keras.backend.floatx()
    try:
        jax.config.update("jax_enable_x64", False)
        keras.backend.set_floatx("float32")
        yield
    finally:
        jax.config.update("jax_enable_x64", x64_enabled)
        keras.backend.set_floatx(floatx)


@contextmanager
def x64_enabled():
    """Context manager for temporarily enabling the `'jax_enable_x64'` JAX
    configuration option, and for temporarily setting the Keras default float
    type to double precision.
    """

    x64_enabled = jax.config.x64_enabled
    floatx = keras.backend.floatx()
    try:
        jax.config.update("jax_enable_x64", True)
        keras.backend.set_floatx("float64")
        yield
    finally:
        jax.config.update("jax_enable_x64", x64_enabled)
        keras.backend.set_floatx(floatx)


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
