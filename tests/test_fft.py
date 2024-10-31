from bt_ocean.fft import dst, idst

import jax.numpy as jnp
from numpy import exp
import pytest

from .test_base import eps
from .test_base import test_precision  # noqa: F401


@pytest.mark.parametrize("N", tuple(range(1, 11)) + (128, 129))
@pytest.mark.parametrize("alpha", (1, exp(0.5), -exp(0.5)))
def test_dst_mode(N, alpha):
    x = jnp.linspace(0, 1, N + 1)
    for n in range(1, N):
        u = alpha * jnp.sin(n * jnp.pi * x)
        u_s = dst(u)
        u_s_ref = jnp.zeros_like(u).at[n].set(alpha)
        assert abs(u_s - u_s_ref).max() < 1.0e3 * eps()


@pytest.mark.parametrize("N", tuple(range(1, 11)) + (128, 129))
@pytest.mark.parametrize("alpha", (1, exp(0.5), -exp(0.5)))
def test_dst_linear_combination(N, alpha):
    x = jnp.linspace(0, 1, N + 1)
    u = jnp.zeros(N + 1)
    for m in range(1, N):
        u = u + alpha * (m % 10 - 5.5) * jnp.sin(m * jnp.pi * x)
    u_s = dst(u)
    u_s_ref = alpha * (jnp.arange(N + 1) % 10 - 5.5)
    u_s_ref = u_s_ref.at[0].set(0)
    u_s_ref = u_s_ref.at[-1].set(0)
    assert abs(u_s - u_s_ref).max() < 1.0e3 * eps()


@pytest.mark.parametrize("N", tuple(range(1, 11)) + (128, 129))
@pytest.mark.parametrize("alpha", (1, exp(0.5), -exp(0.5)))
def test_dst_axis(N, alpha):
    x = jnp.linspace(0, 1, N + 1)
    for n in range(1, N):
        u = jnp.reshape(jnp.outer(alpha * jnp.sin(n * jnp.pi * x), jnp.arange(1, 4)), (1, N + 1, 3))
        u_s = dst(u, axis=1)
        u_s_ref = jnp.zeros_like(u).at[:, n, :].set(alpha * jnp.arange(1, 4))
        assert abs(u_s - u_s_ref).max() < 1.0e3 * eps()


@pytest.mark.parametrize("N", tuple(range(1, 11)) + (128, 129))
@pytest.mark.parametrize("alpha", (1, exp(0.5), -exp(0.5)))
def test_idst_mode(N, alpha):
    x = jnp.linspace(0, 1, N + 1)
    for n in range(1, N):
        u = idst(jnp.zeros_like(x).at[n].set(alpha))
        u_ref = alpha * jnp.sin(n * jnp.pi * x)
        assert abs(u - u_ref).max() < 1.0e3 * eps()
