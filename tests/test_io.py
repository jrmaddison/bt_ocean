from bt_ocean.grid import Grid
from bt_ocean.model import (
    CNAB2Solver, Fields, Parameters, read_fields, read_parameters, read_solver)
from bt_ocean.parameters import parameters

import jax.numpy as jnp
import zarr

from .test_base import test_precision  # noqa: F401


def model_parameters():
    model_parameters = dict(parameters)
    model_parameters["L_x"] = 2 * model_parameters["L_y"]
    model_parameters["N_x"] = 128
    model_parameters["N_y"] = 128
    model_parameters["nu"] = 600
    return Parameters(model_parameters)


def test_parameters_roundtrip(tmp_path):
    parameters = model_parameters()

    filename = tmp_path / "tmp.zarr"
    with zarr.open(filename, "w") as h:
        parameters.write(h)
    with zarr.open(filename, "r") as h:
        input_parameters = read_parameters(h)

    assert set(input_parameters) == set(parameters)
    for key, value in parameters.items():
        assert input_parameters[key] == value


def test_fields_roundtrip(tmp_path):
    L_x, L_y = 2.0, 3.0
    N_x, N_y = 5, 10
    grid = Grid(L_x, L_y, N_x, N_y)
    fields = Fields(grid, {"a", "b"})

    fields["a"] = a = jnp.outer(jnp.arange(N_x + 1, dtype=grid.fdtype),
                                jnp.arange(N_y + 1, dtype=grid.fdtype))
    fields["b"] = b = jnp.outer(jnp.arange(N_x, -1, -1, dtype=grid.fdtype),
                                jnp.arange(N_y + 1, dtype=grid.fdtype))

    filename = tmp_path / "tmp.zarr"
    with zarr.open(filename, "w") as h:
        fields.write(h)
    with zarr.open(filename, "r") as h:
        input_fields = read_fields(h)

    assert input_fields.grid.L_x == L_x
    assert input_fields.grid.L_y == L_y
    assert input_fields.grid.N_x == N_x
    assert input_fields.grid.N_y == N_y
    assert set(input_fields) == {"a", "b"}
    assert (input_fields["a"] == a).all()
    assert (input_fields["b"] == b).all()

    fields.update(input_fields)
    assert set(fields) == {"a", "b"}
    assert (fields["a"] == a).all()
    assert (fields["b"] == b).all()


def test_solver_roundtrip(tmp_path):
    solver = CNAB2Solver(model_parameters())

    for _ in range(5):
        solver.step()

    filename = tmp_path / "tmp.zarr"
    with zarr.open(filename, "w") as h:
        solver.write(h)
    with zarr.open(filename, "r") as h:
        input_solver = read_solver(h)

    assert type(input_solver) is type(solver)
    assert input_solver.n == solver.n

    assert input_solver.grid.L_x == solver.grid.L_x
    assert input_solver.grid.L_y == solver.grid.L_y
    assert input_solver.grid.N_x == solver.grid.N_x
    assert input_solver.grid.N_y == solver.grid.N_y
    assert input_solver.grid.idtype == solver.grid.idtype
    assert input_solver.grid.fdtype == solver.grid.fdtype

    assert input_solver.dealias_grid.L_x == solver.dealias_grid.L_x
    assert input_solver.dealias_grid.L_y == solver.dealias_grid.L_y
    assert input_solver.dealias_grid.N_x == solver.dealias_grid.N_x
    assert input_solver.dealias_grid.N_y == solver.dealias_grid.N_y
    assert input_solver.dealias_grid.idtype == solver.dealias_grid.idtype
    assert input_solver.dealias_grid.fdtype == solver.dealias_grid.fdtype

    assert set(input_solver.parameters) == set(solver.parameters)
    for key, value in solver.parameters.items():
        assert input_solver.parameters[key] == value

    assert set(input_solver.fields) == set(solver.fields)
    for key, value in solver.fields.items():
        assert (input_solver.fields[key] == value).all()

    assert set(input_solver.dealias_fields) == set(solver.dealias_fields)
    for key, value in solver.dealias_fields.items():
        assert (input_solver.dealias_fields[key] == value).all()
