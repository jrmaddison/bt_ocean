from bt_ocean.grid import Grid
from bt_ocean.precision import default_fdtype

import jax.numpy as jnp
from numpy import cbrt
import pytest

from .test_base import eps
from .test_base import test_precision  # noqa: F401


@pytest.mark.parametrize("L_x, L_y", [(1, 1),
                                      (cbrt(3), cbrt(2))])
@pytest.mark.parametrize("N_x, N_y", [(3, 5),
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
    if jnp.finfo(default_fdtype()).bits != 64:
        pytest.skip()
    error_norms = []
    for N in [2048, 4096]:
        N_x = N
        N_y = 3 * N // 2
        grid = Grid(L_x, L_y, N_x, N_y)

        def u0(X, Y):
            return jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * X * Y)

        u = u0(grid.X, grid.Y)
        x = jnp.linspace(-grid.L_x, grid.L_x, 17)
        y = jnp.linspace(-grid.L_y, grid.L_y, 19)
        v = grid.interpolate(u, x, y)
        error_norms.append(abs(v - u0(jnp.outer(x, jnp.ones_like(y)), jnp.outer(jnp.ones_like(x), y))).max())
    error_norms = jnp.array(error_norms)
    orders = jnp.log2(error_norms[:-1] / error_norms[1:])
    print(f"{orders=}")
    assert orders.min() > 1.9
    assert orders.max() < 2.1


@pytest.mark.parametrize("L_x, L_y", [(1, 1),
                                      (cbrt(3), cbrt(2))])
@pytest.mark.parametrize("N_x, N_y", [(3, 5),
                                      (10, 20),
                                      (20, 10),
                                      (32, 64)])
def test_interpolate_non_uniform(L_x, L_y, N_x, N_y):
    if jnp.finfo(default_fdtype()).bits != 64:
        pytest.skip()
    error_norms = []
    for N in [2048, 4096]:
        N_x = N
        N_y = 3 * N // 2
        grid = Grid(L_x, L_y, N_x, N_y)

        def u0(X, Y):
            return jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * X * Y)

        u = u0(grid.X, grid.Y)
        x = jnp.logspace(-1, 0, 17) * grid.L_x
        y = jnp.logspace(-2, 0, 19) * grid.L_y
        v = grid.interpolate(u, x, y)
        error_norms.append(abs(v - u0(jnp.outer(x, jnp.ones_like(y)), jnp.outer(jnp.ones_like(x), y))).max())
    error_norms = jnp.array(error_norms)
    orders = jnp.log2(error_norms[:-1] / error_norms[1:])
    print(f"{orders=}")
    assert orders.min() > 1.9
    assert orders.max() < 2.1
