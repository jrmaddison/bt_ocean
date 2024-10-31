from bt_ocean.grid import Grid
from bt_ocean.inversion import ModifiedHelmholtzSolver, PoissonSolver

import jax.numpy as jnp
from numpy import cbrt, exp
import pytest

from .test_base import eps
from .test_base import test_precision  # noqa: F401


@pytest.mark.parametrize("L_x, L_y", [(1, 1),
                                      (cbrt(3), cbrt(2))])
@pytest.mark.parametrize("N_x, N_y", [(3, 5),
                                      (10, 20),
                                      (20, 10),
                                      (32, 64)])
def test_poisson_solver_residual(L_x, L_y, N_x, N_y):
    solver = PoissonSolver(Grid(L_x, L_y, N_x, N_y))

    X, Y = solver.grid.X, solver.grid.Y
    b_ref = (jnp.sin((jnp.pi * (X + L_x)) / (2 * L_x))
             * jnp.sin((3 * jnp.pi * (Y + L_y)) / (2 * L_y))
             * jnp.exp(X) * jnp.exp(Y))
    u = solver.solve(b_ref)

    b = (solver.grid.D_xx(u) + solver.grid.D_yy(u))[1:-1, 1:-1]
    error_norm = abs(b - b_ref[1:-1, 1:-1]).max() / abs(b).max()
    print(f"{error_norm=}")
    assert error_norm < 1.0e3 * eps()


@pytest.mark.parametrize("L_x, L_y", [(1, 1),
                                      (cbrt(3), cbrt(2))])
@pytest.mark.parametrize("N_x, N_y", [(3, 5),
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
             * jnp.exp(X) * jnp.exp(Y))
    u = solver.solve(b_ref)

    b = (solver.grid.D_xx(u) + solver.grid.D_yy(u) - alpha * u)[1:-1, 1:-1]
    error_norm = abs(b - b_ref[1:-1, 1:-1]).max() / abs(b).max()
    print(f"{error_norm=}")
    assert error_norm < 1.0e3 * eps()
