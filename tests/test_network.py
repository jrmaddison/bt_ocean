import pytest
pytest.importorskip("keras", reason="Keras not available")

import jax.numpy as jnp
import keras
from numpy import sqrt

from bt_ocean.network import KroneckerProduct, Scale

from .test_base import test_precision  # noqa: F401


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
