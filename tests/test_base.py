import jax
import jax.numpy as jnp
import keras
import pytest

from bt_ocean.precision import default_fdtype


__all__ = \
    [
        "eps"
    ]


@pytest.fixture(autouse=True, scope="module",
                params=[{"x64_enabled": False}, {"x64_enabled": True}])
def test_precision(request):
    floatx = keras.backend.floatx()
    x64_enabled = jax.config.x64_enabled
    keras.backend.set_floatx("float64" if request.param["x64_enabled"] else "float32")
    jax.config.update("jax_enable_x64", request.param["x64_enabled"])
    yield
    keras.backend.set_floatx(floatx)
    jax.config.update("jax_enable_x64", x64_enabled)


def eps():
    return jnp.finfo(default_fdtype()).eps
