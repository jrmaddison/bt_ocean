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
    tol = 1.0e-4
    model = CNAB2Solver(model_parameters())

    def update_m(model, m):
        model.fields["Q"] = Q(model.grid) + m

    def forward(Q_1):
        model.initialize()
        model.steady_state_solve((Q_1,), update_m, tol=tol)
        return model.ke()

    J_0, vjp = jax.vjp(forward, jnp.zeros_like(model.fields["Q"]))

    zeta = ((tau_0 * jnp.pi / (D * rho_0 * model.grid.L_y))
            * jnp.cos(3 * model.grid.X / model.grid.L_x) * jnp.cos(5 * model.grid.Y / model.grid.L_y))
    dJ, = vjp(1.0)

    eps = jnp.array((1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5))
    errors = []
    for eps_val in eps:
        J_1 = forward(eps_val * zeta)
        errors.append(abs(J_1 - J_0 - eps_val * (dJ * zeta).sum()))
    errors = jnp.array(errors)
    orders = jnp.log(errors[1:] / errors[:-1]) / jnp.log(eps[1:] / eps[:-1])
    print(f"{orders=}")
    assert orders.min() > 1.99
    assert orders.max() < 2.01
