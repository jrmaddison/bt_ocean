from bt_ocean.grid import Grid

import jax.numpy as jnp
from numpy import cbrt
import pytest
import sympy as sp

from .test_base import eps
from .test_base import test_precision  # noqa: F401


@pytest.mark.parametrize("L_x, L_y", [(1, 1),
                                      (cbrt(3), cbrt(2))])
@pytest.mark.parametrize("N_x, N_y", [(4, 5),
                                      (10, 20),
                                      (20, 10),
                                      (32, 64)])
def test_interpolate_identity(L_x, L_y, N_x, N_y):
    grid = Grid(L_x, L_y, N_x, N_y)

    def u0(X, Y):
        return (jnp.sqrt(2) - jnp.sqrt(3) * X + jnp.sqrt(5) * Y
                - jnp.sqrt(7) * X ** 2 + jnp.sqrt(11) * X * Y - jnp.sqrt(13) * Y ** 2)

    u = u0(grid.X, grid.Y)
    x = grid.x
    y = grid.y
    v = grid.interpolate(u, x, y)
    assert abs(v - u0(jnp.outer(x, jnp.ones_like(y)), jnp.outer(jnp.ones_like(x), y))).max() < 100 * eps()


@pytest.mark.parametrize("L_x, L_y", [(1, 1),
                                      (cbrt(3), cbrt(2))])
def test_interpolate_uniform(L_x, L_y):
    error_norms = []
    for N in [64, 128, 256, 512]:
        N_x = N
        N_y = 3 * N // 2
        grid = Grid(L_x, L_y, N_x, N_y)

        def u0(X, Y):
            return jnp.sin(jnp.pi * X / grid.L_x) * jnp.sin(jnp.pi * X * Y / (grid.L_x * grid.L_y))

        u = u0(grid.X, grid.Y)
        x = jnp.linspace(-grid.L_x, grid.L_x, 17)
        y = jnp.linspace(-grid.L_y, grid.L_y, 19)
        v = grid.interpolate(u, x, y)
        error_norms.append(abs(v - u0(jnp.outer(x, jnp.ones_like(y)), jnp.outer(jnp.ones_like(x), y))).max())
    error_norms = jnp.array(error_norms)
    orders = jnp.log2(error_norms[:-1] / error_norms[1:])
    print(f"{error_norms=}")
    print(f"{orders=}")
    assert orders.min() > 1.97
    assert orders.max() < 2.03


@pytest.mark.parametrize("L_x, L_y", [(1, 1),
                                      (cbrt(3), cbrt(2))])
@pytest.mark.parametrize("N_x, N_y", [(4, 5),
                                      (10, 20),
                                      (20, 10),
                                      (32, 64)])
def test_interpolate_non_uniform(L_x, L_y, N_x, N_y):
    error_norms = []
    for N in [256, 512, 1024]:
        N_x = N
        N_y = 3 * N // 2
        grid = Grid(L_x, L_y, N_x, N_y)

        def u0(X, Y):
            return jnp.sin(jnp.pi * X / grid.L_x) * jnp.sin(jnp.pi * X * Y / (grid.L_x * grid.L_y))

        u = u0(grid.X, grid.Y)
        x = jnp.logspace(-1, 0, 17) * grid.L_x
        y = jnp.logspace(-2, 0, 19) * grid.L_y
        v = grid.interpolate(u, x, y)
        error_norms.append(abs(v - u0(jnp.outer(x, jnp.ones_like(y)), jnp.outer(jnp.ones_like(x), y))).max())
    error_norms = jnp.array(error_norms)
    orders = jnp.log2(error_norms[:-1] / error_norms[1:])
    print(f"{error_norms=}")
    print(f"{orders=}")
    assert orders.min() > 1.97
    assert orders.max() < 2.03


@pytest.mark.parametrize("L_x, L_y", [(1, 1),
                                      (cbrt(3), cbrt(2))])
@pytest.mark.parametrize("N_x, N_y", [(4, 5),
                                      (10, 20),
                                      (20, 10),
                                      (32, 64)])
def test_differentiation(L_x, L_y, N_x, N_y):
    grid = Grid(L_x, L_y, N_x, N_y)

    assert abs(grid.D_x(jnp.ones_like(grid.X))).max() == 0
    assert abs(grid.D_x(grid.X) - jnp.ones_like(grid.X)).max() < 1.0e3 * eps()
    assert abs(grid.D_x(grid.X ** 2) - 2 * grid.X).max() < 1.0e3 * eps()

    assert abs(grid.D_xx(jnp.ones_like(grid.X))).max() == 0
    assert abs(grid.D_xx(grid.X)).max() < 1.0e4 * eps()
    assert abs(grid.D_xx(grid.X ** 2) - 2 * jnp.ones_like(grid.X)).max() < 1.0e5 * eps()
    assert abs(grid.D_xx(grid.X ** 3) - 6 * grid.X).max() < 1.0e5 * eps()

    assert abs(grid.D_y(jnp.ones_like(grid.Y))).max() == 0
    assert abs(grid.D_y(grid.Y) - jnp.ones_like(grid.Y)).max() < 1.0e3 * eps()
    assert abs(grid.D_y(grid.Y ** 2) - 2 * grid.Y).max() < 1.0e3 * eps()

    assert abs(grid.D_yy(jnp.ones_like(grid.Y))).max() == 0
    assert abs(grid.D_yy(grid.Y)).max() < 1.0e4 * eps()
    assert abs(grid.D_yy(grid.Y ** 2) - 2 * jnp.ones_like(grid.Y)).max() < 1.0e5 * eps()
    assert abs(grid.D_yy(grid.Y ** 3) - 6 * grid.Y).max() < 1.0e5 * eps()


@pytest.mark.parametrize("L_x, L_y", [(1, 1),
                                      (cbrt(3), cbrt(2))])
def test_integration(L_x, L_y):
    error_norms = []
    for N in [16, 32, 64, 128]:
        N_x = N
        N_y = 3 * N_x // 2
        grid = Grid(L_x, L_y, N_x, N_y)

        assert abs(grid.integrate(jnp.ones_like(grid.X)) - 4 * grid.L_x * grid.L_y) < 1.0e5 * eps()
        assert abs(grid.integrate(grid.X)) < 1.0e2 * eps()
        assert abs(grid.integrate(grid.Y)) < 1.0e2 * eps()

        error_norms.append(abs(grid.integrate(grid.X ** 2 + jnp.pi * grid.Y ** 2)
                               - (4 / 3) * (grid.L_x ** 3) * grid.L_y
                               - jnp.pi * (4 / 3) * grid.L_x * (grid.L_y ** 3)))
    error_norms = jnp.array(error_norms)
    orders = jnp.log2(error_norms[:-1] / error_norms[1:])
    print(f"{error_norms=}")
    print(f"{orders=}")
    assert orders.min() > 1.9
    assert orders.max() < 2.1


@pytest.mark.parametrize("L_x, L_y", [(1, 1),
                                      (cbrt(3), cbrt(2))])
def test_jacobian(L_x, L_y):
    x = sp.Symbol("x", real=True)
    y = sp.Symbol("y", real=True)
    psi_ref = sp.sin(3 * sp.pi * x / L_x) * sp.sin(5 * sp.pi * y / L_y) * sp.sin(sp.pi * (x + y) / (L_x + L_y))
    q_ref = sp.sin(5 * sp.pi * x / L_x) * sp.sin(7 * sp.pi * y / L_y) * sp.exp(-2 * x * sp.sin(sp.pi * y / L_y) / L_x)
    b_ref = sp.diff(psi_ref, y) * sp.diff(q_ref, x) - sp.diff(psi_ref, x) * sp.diff(q_ref, y)
    psi_ref = sp.lambdify((x, y), psi_ref, modules="jax")
    q_ref = sp.lambdify((x, y), q_ref, modules="jax")
    b_ref = sp.lambdify((x, y), b_ref, modules="jax")
    del x, y

    error_norms = []
    for N in [128, 256, 512, 1024]:
        N_x = N
        N_y = 3 * N_x // 2
        grid = Grid(L_x, L_y, N_x, N_y)
        q = q_ref(grid.X, grid.Y)
        psi = psi_ref(grid.X, grid.Y)

        b = grid.J(q, psi)
        assert abs(b + grid.J(psi, q)).max() < 1.0e4 * eps()
        assert grid.integrate(b) / (4 * grid.L_x * grid.L_y) < 1.0e2 * eps()
        assert grid.integrate(q * b) / (4 * grid.L_x * grid.L_y) < 1.0e2 * eps()
        assert grid.integrate(psi * b) / (4 * grid.L_x * grid.L_y) < 10 * eps()

        error_norms.append(jnp.sqrt(grid.integrate((b - b_ref(grid.X, grid.Y)) ** 2)))
    error_norms = jnp.array(error_norms)
    orders = jnp.log2(error_norms[:-1] / error_norms[1:])
    print(f"{error_norms=}")
    print(f"{orders=}")
    assert orders.min() > 1.98
    assert orders.max() < 2.02
