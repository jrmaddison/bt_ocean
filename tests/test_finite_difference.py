from bt_ocean.finite_difference import (
    difference_coefficients, diff_bounded, diff_periodic)
from bt_ocean.precision import default_fdtype

import jax
import jax.numpy as jnp
import numpy as np
from numpy import exp
import pytest
import sympy as sp


def test_difference_coefficients():
    f = sp.Rational

    def verify(alpha, order, beta):
        assert difference_coefficients(alpha, order) == tuple(map(f, beta))

    # First order derivatives
    verify((0, 1), 1, (-1, 1))
    verify((0, -1), 1, (1, -1))
    verify((0, 2), 1, (-f(1, 2), f(1, 2)))
    verify((-1, 1), 1, (-f(1, 2), f(1, 2)))
    verify((0, -2), 1, (f(1, 2), -f(1, 2)))
    verify((0, 1, 2), 1, (-f(3, 2), 2, -f(1, 2)))
    verify((0, -1, -2), 1, (f(3, 2), -2, f(1, 2)))
    verify((1, 2, 3), 1, (-f(5, 2), 4, -f(3, 2)))
    verify((-1, -2, -3), 1, (f(5, 2), -4, f(3, 2)))

    # Second order derivatives
    verify((-1, 0, 1), 2, (1, -2, 1))
    verify((0, 1, 2, 3), 2, (2, -5, 4, -1))
    verify((0, -1, -2, -3), 2, (2, -5, 4, -1))
    verify((1, 2, 3, 4), 2, (3, -8, 7, -2))
    verify((-1, -2, -3, -4), 2, (3, -8, 7, -2))


@pytest.mark.parametrize("alpha", (1, exp(0.5), -exp(0.5)))
def test_centered_difference_monomials(alpha):
    if default_fdtype() != np.float64 or not jax.config.x64_enabled:
        pytest.skip("float64 not available")

    x = jnp.linspace(-1, 1, 9, dtype=float)
    dx = x[1] - x[0]

    for p in range(17):
        u = alpha * x ** p
        for order in range(7):
            if order > p:
                diff_exact = alpha * jnp.zeros_like(x)
            else:
                diff_exact = alpha * x ** (p - order)
                for i in range(order):
                    diff_exact *= p - i
            for N in range(max(p + 1, order + 1), x.shape[0]):
                diff = diff_bounded(u, dx, order=order, N=N)
                error_norm = abs(diff - diff_exact).max()
                diff_exact_norm = abs(diff_exact).max()
                if diff_exact_norm > 0:
                    assert error_norm < 1e-13 * diff_exact_norm
                else:
                    assert error_norm < 1e-9


def test_centered_difference_convergence():
    if default_fdtype() != np.float64 or not jax.config.x64_enabled:
        pytest.skip("float64 not available")

    P = jnp.arange(7, 12, dtype=int)
    error_norms_1 = jnp.zeros_like(P, dtype=float)
    error_norms_2 = jnp.zeros_like(P, dtype=float)
    for i, p in enumerate(P):
        x = jnp.linspace(0, 1, 2 ** p + 1, dtype=float)
        dx = x[1] - x[0]
        W = jnp.full_like(x, dx)
        W = W.at[0].set(0.5 * dx)
        W = W.at[-1].set(0.5 * dx)
        u = jnp.sin(jnp.pi * x)

        D1_error = (diff_bounded(u, dx, order=1, N=3)
                    - jnp.pi * jnp.cos(jnp.pi * x))
        error_norms_1 = error_norms_1.at[i].set(jnp.sqrt((W * (D1_error ** 2)).sum()))  # noqa: E501
        print(f"{p=:d} {error_norms_1[i]=:.6g}")

        D2_error = (diff_bounded(u, dx, order=2, N=3)
                    + (jnp.pi ** 2) * jnp.sin(jnp.pi * x))
        error_norms_2 = error_norms_2.at[i].set(jnp.sqrt((W * (D2_error ** 2)).sum()))  # noqa: E501
        print(f"{p=:d} {error_norms_2[i]=:.6g}")
    orders_1 = jnp.log2(error_norms_1[:-1] / error_norms_1[1:])
    orders_2 = jnp.log2(error_norms_2[:-1] / error_norms_2[1:])
    print(f"{orders_1=}")
    print(f"{orders_2=}")
    assert orders_1.min() > 2
    assert orders_2.min() > 2


def test_centered_difference_convergence_periodic():
    if default_fdtype() != np.float64 or not jax.config.x64_enabled:
        pytest.skip("float64 not available")

    P = jnp.arange(7, 12, dtype=int)
    error_norms_1 = jnp.zeros_like(P, dtype=float)
    error_norms_2 = jnp.zeros_like(P, dtype=float)
    for i, p in enumerate(P):
        x = jnp.linspace(0, 1, 2 ** p + 1, dtype=float)[:-1]
        dx = x[1] - x[0]
        W = jnp.full_like(x, dx)
        u = jnp.sin(2 * jnp.pi * x)

        D1_error = (diff_periodic(u, dx, order=1, N=3)
                    - 2 * jnp.pi * jnp.cos(2 * jnp.pi * x))
        error_norms_1 = error_norms_1.at[i].set(jnp.sqrt((W * (D1_error ** 2)).sum()))  # noqa: E501
        print(f"{p=:d} {error_norms_1[i]=:.6g}")

        D2_error = (diff_periodic(u, dx, order=2, N=3)
                    + 4 * (jnp.pi ** 2) * jnp.sin(2 * jnp.pi * x))
        error_norms_2 = error_norms_2.at[i].set(jnp.sqrt((W * (D2_error ** 2)).sum()))  # noqa: E501
        print(f"{p=:d} {error_norms_2[i]=:.6g}")
    orders_1 = jnp.log2(error_norms_1[:-1] / error_norms_1[1:])
    orders_2 = jnp.log2(error_norms_2[:-1] / error_norms_2[1:])
    print(f"{orders_1=}")
    print(f"{orders_2=}")
    assert orders_1.min() > 1.99
    assert orders_2.min() > 1.99
