from bt_ocean.grid import Grid
from bt_ocean.inversion import ModifiedHelmholtzSolver, PoissonSolver

import jax.numpy as jnp
from numpy import cbrt, exp
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
def test_poisson_solver_residual(L_x, L_y, N_x, N_y):
    solver = PoissonSolver(Grid(L_x, L_y, N_x, N_y))

    X, Y = solver.grid.X, solver.grid.Y
    b_ref = (jnp.sin((jnp.pi * (X + L_x)) / (2 * L_x))
             * jnp.sin((3 * jnp.pi * (Y + L_y)) / (2 * L_y))
             * jnp.exp(X / L_x) * jnp.exp(Y / L_y))
    u = solver.solve(b_ref)

    b = (solver.grid.D_xx(u, boundary=False) + solver.grid.D_yy(u, boundary=False))[1:-1, 1:-1]
    error_norm = abs(b - b_ref[1:-1, 1:-1]).max() / abs(b).max()
    print(f"{error_norm=}")
    assert error_norm < 1.0e3 * eps()


@pytest.mark.parametrize("L_x, L_y", [(1, 1),
                                      (cbrt(3), cbrt(2))])
def test_poisson_solver_convergence(L_x, L_y):
    x = sp.Symbol("x", real=True)
    y = sp.Symbol("y", real=True)
    u_ref = sp.sin(3 * sp.pi * x / L_x) * sp.sin(5 * sp.pi * y / L_y) * sp.sin(sp.pi * (x + y) / (L_x + L_y))
    b_ref = sp.diff(u_ref, x, 2) + sp.diff(u_ref, y, 2)
    u_ref = sp.lambdify((x, y), u_ref, modules="jax")
    b_ref = sp.lambdify((x, y), b_ref, modules="jax")
    del x, y

    error_norms = []
    for N in [32, 64, 128, 256]:
        N_x = N
        N_y = 3 * N_x // 2
        solver = PoissonSolver(
            Grid(L_x, L_y, N_x, N_y))

        X, Y = solver.grid.X, solver.grid.Y
        u = solver.solve(b_ref(X, Y))
        error_norms.append(jnp.sqrt(solver.grid.integrate((u - u_ref(X, Y)) ** 2)))
    error_norms = jnp.array(error_norms)
    orders = jnp.log2(error_norms[:-1] / error_norms[1:])
    print(f"{error_norms=}")
    print(f"{orders=}")
    assert orders.min() > 1.99
    assert orders.max() < 2.03


@pytest.mark.parametrize("L_x, L_y", [(1, 1),
                                      (cbrt(3), cbrt(2))])
@pytest.mark.parametrize("N_x, N_y", [(4, 5),
                                      (10, 20),
                                      (20, 10),
                                      (32, 64)])
@pytest.mark.parametrize("alpha", (0.5, 1, 2, exp(-0.5)))
def test_modified_helmholtz_solver_residual(L_x, L_y, N_x, N_y, alpha):
    solver = ModifiedHelmholtzSolver(
        Grid(L_x, L_y, N_x, N_y), alpha=alpha)

    X, Y = solver.grid.X, solver.grid.Y
    b_ref = (jnp.sin((jnp.pi * (X + L_x)) / (2 * L_x))
             * jnp.sin((3 * jnp.pi * (Y + L_y)) / (2 * L_y))
             * jnp.exp(X / L_x) * jnp.exp(Y / L_y))
    u = solver.solve(b_ref)

    b = (solver.grid.D_xx(u, boundary=False) + solver.grid.D_yy(u, boundary=False) - alpha * u)[1:-1, 1:-1]
    error_norm = abs(b - b_ref[1:-1, 1:-1]).max() / abs(b).max()
    print(f"{error_norm=}")
    assert error_norm < 1.0e3 * eps()


@pytest.mark.parametrize("L_x, L_y", [(1, 1),
                                      (cbrt(3), cbrt(2))])
def test_modified_helmholtz_solver_convergence(L_x, L_y):
    alpha = jnp.exp(-0.5)

    x = sp.Symbol("x", real=True)
    y = sp.Symbol("y", real=True)
    u_ref = sp.sin(3 * sp.pi * x / L_x) * sp.sin(5 * sp.pi * y / L_y) * sp.sin(sp.pi * (x + y) / (L_x + L_y))
    b_ref = sp.diff(u_ref, x, 2) + sp.diff(u_ref, y, 2) - alpha * u_ref
    u_ref = sp.lambdify((x, y), u_ref, modules="jax")
    b_ref = sp.lambdify((x, y), b_ref, modules="jax")
    del x, y

    error_norms = []
    for N in [32, 64, 128, 256]:
        N_x = N
        N_y = 3 * N_x // 2
        solver = ModifiedHelmholtzSolver(
            Grid(L_x, L_y, N_x, N_y), alpha=alpha)

        X, Y = solver.grid.X, solver.grid.Y
        u = solver.solve(b_ref(X, Y))
        error_norms.append(jnp.sqrt(solver.grid.integrate((u - u_ref(X, Y)) ** 2)))
    error_norms = jnp.array(error_norms)
    orders = jnp.log2(error_norms[:-1] / error_norms[1:])
    print(f"{error_norms=}")
    print(f"{orders=}")
    assert orders.min() > 1.99
    assert orders.max() < 2.03
