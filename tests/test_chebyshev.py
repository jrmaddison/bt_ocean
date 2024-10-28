from bt_ocean.chebyshev import Chebyshev, InterpolationMethod

import jax.numpy as jnp
from numpy import exp
import pytest

from .test_base import eps
from .test_base import test_precision  # noqa: F401


@pytest.mark.parametrize("alpha", (1, exp(0.5), -exp(0.5)))
@pytest.mark.parametrize("N", tuple(range(1, 11)) + (128, 129))
def test_chebyshev_basis(alpha, N):
    cheb = Chebyshev(N)

    x0 = jnp.ones_like(cheb.x)
    x1 = cheb.x

    for n in range(N + 1):
        if n == 0:
            u_ref = alpha * x0
        elif n == 1:
            u_ref = alpha * x1
        else:
            x1, x0 = 2 * cheb.x * x1 - x0, x1
            u_ref = alpha * x1
        u_c = cheb.to_cheb(u_ref)
        u_c_ref = jnp.zeros_like(u_c)
        u_c_ref = u_c_ref.at[n].set(alpha)
        assert abs(u_c - u_c_ref).max() < 100 * eps()
        u = cheb.from_cheb(u_c)
        assert abs(u - u_ref).max() < 100 * eps()


@pytest.mark.parametrize("N", tuple(range(3, 11)) + (128, 129))
@pytest.mark.parametrize("interpolation_method", [InterpolationMethod.BARYCENTRIC,
                                                  InterpolationMethod.CLENSHAW])
def test_chebyshev_interpolation_identity(N, interpolation_method):
    cheb = Chebyshev(N)

    def u0(x):
        return jnp.sqrt(2) - jnp.sqrt(3) * x + jnp.sqrt(5) * x ** 2 - jnp.sqrt(7) * x ** 3

    def u1(x):
        return -jnp.sqrt(11) + jnp.sqrt(13) * x - jnp.sqrt(17) * x ** 2 + jnp.sqrt(19) * x ** 3

    u = jnp.vstack((u0(cheb.x), u1(cheb.x))).T
    x = cheb.x

    v = cheb.interpolate(u, x, axis=0, interpolation_method=interpolation_method)
    assert abs(v[:, 0] - u0(x)).max() < 100 * eps()
    assert abs(v[:, 1] - u1(x)).max() < 100 * eps()


@pytest.mark.parametrize("N", tuple(range(3, 11)) + (128, 129))
@pytest.mark.parametrize("interpolation_method", [InterpolationMethod.BARYCENTRIC,
                                                  InterpolationMethod.CLENSHAW])
def test_chebyshev_interpolation(N, interpolation_method):
    cheb = Chebyshev(N)

    def u0(x):
        return jnp.sqrt(2) - jnp.sqrt(3) * x + jnp.sqrt(5) * x ** 2 - jnp.sqrt(7) * x ** 3

    def u1(x):
        return -jnp.sqrt(11) + jnp.sqrt(13) * x - jnp.sqrt(17) * x ** 2 + jnp.sqrt(19) * x ** 3

    u = jnp.vstack((u0(cheb.x), u1(cheb.x))).T
    x = jnp.array((-1 / jnp.pi, 2 / jnp.pi))

    v = cheb.interpolate(u, x, axis=0, interpolation_method=interpolation_method)
    assert abs(v[:, 0] - u0(x)).max() < 100 * eps()
    assert abs(v[:, 1] - u1(x)).max() < 100 * eps()


@pytest.mark.parametrize("alpha", (1, exp(0.5), -exp(0.5)))
@pytest.mark.parametrize("N", tuple(range(1, 11)))
def test_chebyshev_differentiation_matrix_monomial(alpha, N):
    cheb = Chebyshev(N)
    x = cheb.x

    for n in range(N + 1):
        v = alpha * x ** n
        D_v = cheb.D @ v
        if n == 0:
            D_v_ref = jnp.zeros_like(x)
        else:
            D_v_ref = alpha * n * (x ** (n - 1))
        assert abs(D_v - D_v_ref).max() < 100 * eps()


@pytest.mark.parametrize("alpha", (1, exp(0.5), -exp(0.5)))
@pytest.mark.parametrize("N", tuple(range(1, 11)) + (128,))
def test_clenshaw_curtis_quadrature(alpha, N):
    cheb = Chebyshev(N)
    x = cheb.x
    w = cheb.w

    for n in range(N + 1):
        v = alpha * x ** n
        integral = w @ v
        integral_ref = alpha * (2 / (n + 1)) * int(n % 2 == 0)
        assert abs(integral - integral_ref) < 10 * eps()
