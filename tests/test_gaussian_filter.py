import jax
import jax.numpy as jnp
import pytest
from scipy.ndimage import gaussian_filter as sp_gaussian_filter

from bt_ocean.gaussian_filter import gaussian_filter_1d, gaussian_filter


@pytest.mark.parametrize("sigma", (1, 1.5, 2, 4, 6, 8))
@pytest.mark.parametrize("mode", ("reflect", "constant", "nearest", "mirror", "wrap"))
@pytest.mark.parametrize("truncate", (4, 3.5, 3, 2))
@pytest.mark.parametrize("axis", (0, 1))
def test_gaussian_filter_1d(sigma, mode, truncate, axis):
    if not jax.config.x64_enabled:
        pytest.skip("float64 not available")

    x = jnp.linspace(-2, 1.5, 29)
    y = jnp.linspace(-3.5, 2.5, 30)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    u = jnp.exp(X) * jnp.sinh(jnp.pi * Y)

    u_bar = gaussian_filter_1d(
        u, sigma, mode=mode, cval=-3,
        truncate=truncate, axis=axis)
    u_bar_ref = sp_gaussian_filter(
        u, sigma, mode=mode, cval=-3,
        radius=round(sigma * truncate), axes=axis)
    u_bar_error_norm = abs(u_bar - u_bar_ref).max()
    print(f"{u_bar_error_norm=}")
    assert u_bar_error_norm < 1e-10


@pytest.mark.parametrize("sigma", (1, 1.5, 2, 4, 6, 8))
@pytest.mark.parametrize("mode", ("reflect", "constant", "nearest", "mirror", "wrap"))
@pytest.mark.parametrize("truncate", (4, 3.5, 3, 2))
def test_gaussian_filter(sigma, mode, truncate):
    if not jax.config.x64_enabled:
        pytest.skip("float64 not available")

    x = jnp.linspace(-2, 1.5, 29)
    y = jnp.linspace(-3.5, 2.5, 30)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    u = jnp.exp(X) * jnp.sinh(jnp.pi * Y)

    u_bar = gaussian_filter(
        u, sigma, mode=mode, cval=-3,
        truncate=truncate)
    u_bar_ref = sp_gaussian_filter(
        u, sigma, mode=mode, cval=-3,
        radius=round(sigma * truncate))
    u_bar_error_norm = abs(u_bar - u_bar_ref).max()
    print(f"{u_bar_error_norm=}")
    assert u_bar_error_norm < 1e-9
