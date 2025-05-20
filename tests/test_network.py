import jax.numpy as jnp
import keras
import numpy as np
from numpy import sqrt
import pytest

from bt_ocean.model import CNAB2Solver, Parameters
from bt_ocean.network import Dynamics, KroneckerProduct, Scale
from bt_ocean.parameters import parameters, Q
from bt_ocean.precision import default_fdtype


pytestmark = pytest.mark.skipif(
    keras.backend.backend() != "jax",
    reason="Require Keras with the JAX backend")


@pytest.fixture(scope="module", autouse=True)
def set_floatx():
    floatx = keras.backend.floatx()
    try:
        keras.backend.set_floatx(
            {np.float32: "float32", np.float64: "float64"}[default_fdtype()])
        yield
    finally:
        keras.backend.set_floatx(floatx)


def model_parameters():
    n_hour = 1
    model_parameters = dict(parameters)
    model_parameters["dt"] = 3600 / n_hour
    model_parameters["N_x"] = 32
    model_parameters["N_y"] = 32
    model_parameters["nu"] = 1.0e5
    return Parameters(model_parameters)


@pytest.mark.parametrize("alpha", [-sqrt(2), sqrt(3)])
def test_scale_roundtrip(tmp_path, alpha):
    input_layer = keras.layers.Input((3, 2))
    scale_layer = Scale()
    output_layer = scale_layer(input_layer)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    w = [jnp.array((alpha,), dtype=scale_layer.get_weights()[0].dtype)]
    scale_layer.set_weights(w)
    assert isinstance(scale_layer, Scale)
    assert len(scale_layer.get_weights()) == 1
    assert scale_layer.get_weights()[0].shape == (1,)
    assert scale_layer.get_weights()[0][0] == w

    model.save(tmp_path / "tmp.keras")
    model = keras.models.load_model(tmp_path / "tmp.keras")

    _, scale_layer = model.layers
    assert isinstance(scale_layer, Scale)
    assert len(scale_layer.get_weights()) == 1
    assert scale_layer.get_weights()[0].shape == (1,)
    assert scale_layer.get_weights()[0][0] == w


@pytest.mark.parametrize("activation", [None, "relu"])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_kronecker_product_roundtrip(tmp_path, activation, symmetric, bias):
    input_layer = keras.layers.Input((3, 2))
    kp_layer = KroneckerProduct((3, 2), (7, 5), activation=activation, symmetric=symmetric, bias=bias)
    output_layer = kp_layer(input_layer)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    w = kp_layer.get_weights()
    assert len(w) == 3 if bias else 2
    w = tuple(map(jnp.array,
                  (jnp.reshape(jnp.arange(jnp.prod(jnp.array(w_i.shape)), dtype=w_i.dtype), w_i.shape)
                   for w_i in w)))
    kp_layer.set_weights(w)
    assert isinstance(kp_layer, KroneckerProduct)
    assert kp_layer._KroneckerProduct__activation_arg == activation
    assert kp_layer._KroneckerProduct__symmetric == symmetric
    assert kp_layer._KroneckerProduct__bias == bias
    assert len(kp_layer.get_weights()) == len(w)
    for w_i, w_j in zip(kp_layer.get_weights(), w):
        assert w_i.shape == w_j.shape
        assert w_i.dtype == w_j.dtype
        assert (w_i == w_j).all()

    model.save(tmp_path / "tmp.keras")
    model = keras.models.load_model(tmp_path / "tmp.keras")

    _, kp_layer = model.layers
    assert isinstance(kp_layer, KroneckerProduct)
    assert kp_layer._KroneckerProduct__activation_arg == activation
    assert kp_layer._KroneckerProduct__symmetric == symmetric
    assert kp_layer._KroneckerProduct__bias == bias
    assert len(kp_layer.get_weights()) == len(w)
    for w_i, w_j in zip(kp_layer.get_weights(), w):
        assert w_i.shape == w_j.shape
        assert w_i.dtype == w_j.dtype
        assert (w_i == w_j).all()


def test_dynamics_roundtrip(tmp_path):
    model = CNAB2Solver(model_parameters())
    model.fields["Q"] = Q(model.grid)
    model.steps(5)

    Q_input_layer = keras.layers.Input((model.grid.N_x + 1, model.grid.N_y + 1))
    Q_network = keras.models.Model(inputs=Q_input_layer, outputs=keras.layers.Add()((Q_input_layer,)))

    # Used to test that Q_callback is being called, although technically
    # Q_callback is not allowed to have any side effects other than to modify
    # the dynamics argument
    n_calls = 0

    @Dynamics.register_update("test_dynamics_roundtrip_Q_callback")
    def Q_callback(dynamics, Q_network):
        nonlocal n_calls
        n_calls += 1

    dynamics_layer = Dynamics(model, Q_callback, Q_network)
    dynamics_input_layer = keras.layers.Input((model.grid.N_x + 1, model.grid.N_y + 1))
    dynamics_network = keras.models.Model(inputs=dynamics_input_layer, outputs=dynamics_layer(dynamics_input_layer))

    assert n_calls == 0
    dynamics_network(jnp.zeros((1, model.grid.N_x + 1, model.grid.N_y + 1)))
    assert n_calls == 1

    dynamics_network.save(tmp_path / "tmp.keras")
    dynamics_network = keras.models.load_model(tmp_path / "tmp.keras")

    input_model = dynamics_network.layers[1]._Dynamics__dynamics

    assert type(input_model) is type(model)
    assert input_model.n == 0

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
    for key in model.prescribed_field_keys:
        assert (input_model.fields[key] == model.fields[key]).all()

    dynamics_network(jnp.zeros((1, model.grid.N_x + 1, model.grid.N_y + 1)))
    assert n_calls == 2
