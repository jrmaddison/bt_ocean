from bt_ocean.grid import Grid
from bt_ocean.model import CNAB2Solver, Parameters
from bt_ocean.parameters import parameters, rho_0, D, tau_0, Q
from bt_ocean.precision import x64_enabled

import jax
import jax.numpy as jnp
import pytest


def model_parameters():
    n_hour = 1
    model_parameters = dict(parameters)
    model_parameters["dt"] = 3600 / n_hour
    model_parameters["N_x"] = 32
    model_parameters["N_y"] = 32
    model_parameters["nu"] = 1.0e5
    return Parameters(model_parameters)


@pytest.mark.parametrize("tol", [1.0e-2, 1.0e-3, 1.0e-4])
@x64_enabled()
def test_steady_state(tol):
    model = CNAB2Solver(model_parameters())
    model.fields["Q"] = Q(model.grid)

    zeta_n = model.fields["zeta"]
    model.step()
    zeta_np1 = model.fields["zeta"]
    assert abs(zeta_np1 - zeta_n).max() > tol * abs(zeta_np1).max()

    model.initialize()
    model.steady_state_solve(tol=tol)
    print(f"{model.n=}")

    zeta_n = model.fields["zeta"]
    model.step()
    zeta_np1 = model.fields["zeta"]
    assert abs(zeta_np1 - zeta_n).max() <= tol * abs(zeta_np1).max()


@x64_enabled()
def test_steady_state_autodiff():
    tol = 1.0e-10
    parameters = model_parameters()
    grid = Grid(parameters["L_x"], parameters["L_y"], parameters["N_x"], parameters["N_y"])

    def forward(Q):
        model = CNAB2Solver(parameters)
        model.fields["Q"] = Q
        model.steady_state_solve(tol=tol)
        return jnp.tensordot(model.fields["zeta"], model.grid.W)

    J_0, vjp = jax.vjp(forward, Q(grid))

    zeta = ((tau_0 * jnp.pi / (D * rho_0 * grid.L_y))
            * jnp.exp(grid.X / grid.L_x) * jnp.exp(grid.Y / grid.L_y))
    dJ, = vjp(1.0)

    eps = jnp.array((0.01, 0.005, 0.002, 0.001))
    errors = []
    for eps_val in eps:
        J_1 = forward(Q(grid) + eps_val * zeta)
        errors.append(abs(J_1 - J_0 - eps_val * jnp.tensordot(dJ, zeta)))
    errors = jnp.array(errors)
    orders = jnp.log(errors[1:] / errors[:-1]) / jnp.log(eps[1:] / eps[:-1])
    print(f"{errors=}")
    print(f"{orders=}")
    assert orders.min() > 1.99
    assert orders.max() < 2.02
