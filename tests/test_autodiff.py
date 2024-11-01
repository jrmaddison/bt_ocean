from bt_ocean.model import CNAB2Solver, Parameters
from bt_ocean.parameters import parameters, rho_0, D, tau_0, Q
from bt_ocean.precision import x64_enabled

import jax
import jax.numpy as jnp
import pytest


def model_parameters():
    model_parameters = dict(parameters)
    model_parameters["N_x"] = 128
    model_parameters["N_y"] = 128
    model_parameters["nu"] = 600
    return Parameters(model_parameters)


@pytest.mark.parametrize("N", tuple(range(1, 5)))
@x64_enabled()
def test_cnab2_autodiff_jvp(N):
    model = CNAB2Solver(model_parameters())

    def forward(Q_1):
        model.initialize()
        model.fields["Q"] = Q(model.grid) + Q_1
        model.steps(N)
        return model.ke()

    J_0, jvp = jax.linearize(forward, jnp.zeros_like(model.fields["Q"]))

    zeta = ((tau_0 * jnp.pi / (D * rho_0 * model.grid.L_y))
            * jnp.cos(3 * model.grid.X / model.grid.L_x) * jnp.cos(5 * model.grid.Y / model.grid.L_y))
    dJ = jvp(zeta)

    eps = jnp.array((1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5))
    error_norms = []
    for eps_val in eps:
        J_1 = forward(eps_val * zeta)
        error_norms.append(abs(J_1 - J_0 - eps_val * dJ))
    error_norms = jnp.array(error_norms)
    orders = jnp.log(error_norms[1:] / error_norms[:-1]) / jnp.log(eps[1:] / eps[:-1])
    print(f"{error_norms=}")
    print(f"{orders=}")
    assert orders.min() > 1.99
    assert orders.max() < 2.01


@pytest.mark.parametrize("N", tuple(range(1, 5)))
@x64_enabled()
def test_cnab2_autodiff_vjp(N):
    model = CNAB2Solver(model_parameters())

    def forward(Q_1):
        model.initialize()
        model.fields["Q"] = Q(model.grid) + Q_1
        model.steps(N)
        return model.ke()

    J_0, vjp = jax.vjp(forward, jnp.zeros_like(model.fields["Q"]))

    zeta = ((tau_0 * jnp.pi / (D * rho_0 * model.grid.L_y))
            * jnp.cos(3 * model.grid.X / model.grid.L_x) * jnp.cos(5 * model.grid.Y / model.grid.L_y))
    dJ, = vjp(1.0)

    eps = jnp.array((1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5))
    error_norms = []
    for eps_val in eps:
        J_1 = forward(eps_val * zeta)
        error_norms.append(abs(J_1 - J_0 - eps_val * jnp.tensordot(dJ, zeta)))
    error_norms = jnp.array(error_norms)
    orders = jnp.log(error_norms[1:] / error_norms[:-1]) / jnp.log(eps[1:] / eps[:-1])
    print(f"{error_norms=}")
    print(f"{orders=}")
    assert orders.min() > 1.99
    assert orders.max() < 2.01
