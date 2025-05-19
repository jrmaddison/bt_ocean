from bt_ocean.grid import Grid
from bt_ocean.model import CNAB2Solver, Fields, Parameters, Solver
from bt_ocean.parameters import parameters, Q
from bt_ocean.precision import default_fdtype

try:
    import h5py
except ModuleNotFoundError:
    h5py = None
import jax
import jax.numpy as jnp
import numpy as np
import pytest
try:
    import zarr
except ModuleNotFoundError:
    zarr = None

from contextlib import contextmanager


def model_parameters():
    model_parameters = dict(parameters)
    model_parameters["L_x"] = 2 * model_parameters["L_y"]
    model_parameters["N_x"] = 128
    model_parameters["N_y"] = 128
    model_parameters["nu"] = 600
    return Parameters(model_parameters)


@contextmanager
def h5py_File(filename, mode):
    if h5py is None:
        pytest.skip("h5py not available")
    with h5py.File(filename, mode) as h:
        yield h


@contextmanager
def zarr_File(filename, mode):
    if zarr is None:
        pytest.skip("zarr not available")
    yield zarr.open(filename, mode=mode)


@pytest.mark.parametrize("File", [h5py_File, zarr_File])
def test_parameters_roundtrip(tmp_path, File):
    parameters = model_parameters()

    filename = tmp_path / "tmp"
    with File(filename, "w") as h:
        parameters.write(h)
    with File(filename, "r") as h:
        input_parameters = Parameters.read(h)

    assert set(input_parameters) == set(parameters)
    for key, value in parameters.items():
        assert input_parameters[key] == value


@pytest.mark.parametrize("File", [h5py_File, zarr_File])
def test_fields_roundtrip(tmp_path, File):
    L_x, L_y = 2.0, 3.0
    N_x, N_y = 5, 10
    grid = Grid(L_x, L_y, N_x, N_y)
    fields = Fields(grid, {"a", "b"})

    fields["a"] = a = jnp.outer(jnp.arange(N_x + 1, dtype=grid.fdtype),
                                jnp.arange(N_y + 1, dtype=grid.fdtype))
    fields["b"] = b = jnp.outer(jnp.arange(N_x, -1, -1, dtype=grid.fdtype),
                                jnp.arange(N_y + 1, dtype=grid.fdtype))

    filename = tmp_path / "tmp"
    with File(filename, "w") as h:
        fields.write(h)
    with File(filename, "r") as h:
        input_fields = Fields.read(h)

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


@pytest.mark.parametrize("File", [h5py_File, zarr_File])
def test_solver_roundtrip(tmp_path, File):
    model = CNAB2Solver(model_parameters())
    model.fields["Q"] = Q(model.grid)
    model.steps(5)

    filename = tmp_path / "tmp"
    with File(filename, "w") as h:
        model.write(h)
    with File(filename, "r") as h:
        input_model = Solver.read(h)

    assert type(input_model) is type(model)
    assert input_model.n == model.n

    assert input_model.grid.L_x == model.grid.L_x
    assert input_model.grid.L_y == model.grid.L_y
    assert input_model.grid.N_x == model.grid.N_x
    assert input_model.grid.N_y == model.grid.N_y
    assert input_model.grid.idtype == model.grid.idtype
    assert input_model.grid.fdtype == model.grid.fdtype

    assert set(input_model.parameters) == set(model.parameters)
    for key, value in model.parameters.items():
        assert input_model.parameters[key] == value

    assert set(input_model.fields) == set(model.fields)
    for key, value in model.fields.items():
        assert (input_model.fields[key] == value).all()


@pytest.mark.parametrize("File", [h5py_File, zarr_File])
def test_solver_roundtrip_precision_change(tmp_path, File):
    if default_fdtype() != np.float64 or not jax.config.x64_enabled:
        pytest.skip("float64 not available")

    model = CNAB2Solver(model_parameters(), idtype=np.int32, fdtype=np.float32)
    model.fields["Q"] = Q(model.grid)
    model.steps(5)

    filename = tmp_path / "tmp"
    with File(filename, "w") as h:
        model.write(h)
    with File(filename, "r") as h:
        input_model = Solver.read(h)

    assert type(input_model) is type(model)
    assert input_model.n == model.n

    assert input_model.grid.L_x == model.grid.L_x
    assert input_model.grid.L_y == model.grid.L_y
    assert input_model.grid.N_x == model.grid.N_x
    assert input_model.grid.N_y == model.grid.N_y
    assert input_model.grid.idtype == model.grid.idtype
    assert input_model.grid.fdtype == model.grid.fdtype

    assert set(input_model.parameters) == set(model.parameters)
    for key, value in model.parameters.items():
        assert input_model.parameters[key] == value

    assert set(input_model.fields) == set(model.fields)
    for key, value in model.fields.items():
        assert (input_model.fields[key] == value).all()
