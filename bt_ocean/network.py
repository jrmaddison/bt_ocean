"""Keras utilities.
"""

import jax
import keras
import numpy as np

from itertools import chain
from operator import itemgetter

from .precision import default_fdtype

__all__ = \
    [
        "Scale",

        "KroneckerProduct",
        "kronecker_product_network",

        "Dynamics"
    ]


@keras.saving.register_keras_serializable(package="bt_ocean_scale")
class Scale(keras.layers.Layer):
    """A layer which multiplies by a constant trainable weight.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__alpha = self.add_weight(shape=(1,), dtype=self.dtype)

    def call(self, inputs):
        return self.__alpha[0] * inputs

    def build(self, input_shape):
        pass


@keras.saving.register_keras_serializable(package="bt_ocean_kronecker_product")
class KroneckerProduct(keras.layers.Layer):
    """A layer where the weights matrix has Kronecker product structure.

    Parameters
    ----------

    shape_a : tuple[Integral, Integral]
        Defines the input shape.
    shape_b : tuple[Integral, Integral]
        Defines the output shape.
    symmetric : bool
        Whether to enforce reflectional symmetry.
    bias : bool
        Whether to include a bias term.
    activation
        Defines the activation function.
    dtype : type
        Floating point scalar data type.
    kwargs : dict
        Passed to the base class constructor.
    """

    def __init__(self, shape_a, shape_b, activation, *, symmetric=False, bias=True, dtype=None,
                 **kwargs):
        if dtype is None:
            dtype = default_fdtype()

        super().__init__(dtype=dtype, **kwargs)
        self.__shape_a = shape_a = N_i_a, N_j_a = tuple(shape_a)
        self.__shape_b = shape_b = N_i_b, N_j_b = tuple(shape_b)
        self.__A_i = self.add_weight(shape=(N_i_b, N_i_a), dtype=self.dtype)
        self.__A_j = self.add_weight(shape=(N_j_a, N_j_b), dtype=self.dtype)
        self.__symmetric = symmetric
        self.__bias = bias
        if bias:
            self.__b = self.add_weight(shape=shape_b, dtype=self.dtype)
        else:
            self.__b = None
        self.__activation_arg = activation
        self.__activation = None if activation is None else keras.layers.Activation(activation, dtype=self.dtype)

    def call(self, inputs):
        if self.__symmetric:
            A_i = 0.5 * (self.__A_i + keras.ops.flip(keras.ops.flip(self.__A_i, axis=0), axis=1))
            A_j = 0.5 * (self.__A_j + keras.ops.flip(keras.ops.flip(self.__A_j, axis=0), axis=1))
        else:
            A_i = self.__A_i
            A_j = self.__A_j
        outputs = A_i @ inputs @ A_j
        if self.__b is not None:
            outputs = outputs + self.__b
        if self.__activation is not None:
            outputs = self.__activation(outputs)
        return outputs

    def get_config(self):
        def update(config, d):
            config = dict(config)
            for key, value in d.items():
                if key in config:
                    raise KeyError(f"key '{key}' already defined")
                config[key] = value
            return config

        return update(super().get_config(),
                      {"shape_a": self.__shape_a,
                       "shape_b": self.__shape_b,
                       "activation": self.__activation_arg,
                       "symmetric": self.__symmetric,
                       "bias": self.__bias})

    def build(self, input_shape):
        if tuple(input_shape)[-2:] != self.__shape_a:
            raise ValueError("Invalid shape")


def kronecker_product_network(
        N_i, N_j, *, factor=2, min_size=4, activation="elu",
        final_activation=None, layers_per_level=0, skip=False, dtype=None):
    """A U-shaped network constructed using :class:`.KroneckerProduct` layers,
    down-scaling the number of rows and columns by a given factor between each
    level.

    Parameters
    ----------

    N_i : Integral
        The number of rows.
    N_j : Integral
        The number of columns.
    factor : Real
        Downscaling factor.
    min_size : Integral
        The minimum number of rows or columns for the deepest level.
    activation
        Defines the activation functions applied in each
        :class:`.KroneckerProduct`.
    final_activation
        Defines the activation function for the final layer of the network.
    layers_per_level : Integral
        The number of :class:`.KroneckerProduct` layers within each level.
    skip : bool
        Whether to enable additive skip connections.
    dtype : type
        Floating point scalar data type.

    Returns
    -------

    object
        The network.
    """

    shapes = [(N_i, N_j)]
    p = factor
    while min(N_i, N_j) // p >= min_size:
        shapes.append((N_i // p, N_j // p))
        p *= factor
    if len(shapes) < 2:
        raise ValueError("Size too small")

    input_layer = keras.layers.Input((N_i, N_j))
    output_layer = input_layer

    down_layers = []
    for i in range(len(shapes) - 1):
        for _ in range(layers_per_level):
            output_layer = KroneckerProduct(
                shapes[i], shapes[i], dtype=dtype,
                activation=activation)(output_layer)
        down_layers.append(output_layer)
        output_layer = KroneckerProduct(
            shapes[i], shapes[i + 1], dtype=dtype,
            activation=activation)(output_layer)
    down_layers.append(output_layer)

    for _ in range(layers_per_level):
        output_layer = KroneckerProduct(
            shapes[-1], shapes[-1], dtype=dtype,
            activation=activation)(output_layer)

    for i in range(len(shapes) - 2, -1, -1):
        output_layer = KroneckerProduct(
            shapes[i + 1], shapes[i], dtype=dtype,
            activation=activation if i != 0 or layers_per_level > 0 else final_activation)(output_layer)
        if skip:
            output_layer = keras.layers.Add()((output_layer, down_layers[i]))
        for j in range(layers_per_level):
            output_layer = KroneckerProduct(
                shapes[i], shapes[i], dtype=dtype,
                activation=activation if i != 0 or j < layers_per_level - 1 else final_activation)(output_layer)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


class Dynamics(keras.layers.Layer):
    """Defines a layer consisting of a dynamical core with a neural network
    parameterized forcing.

    Parameters
    ----------

    dynamics : :class:`.Solver`
        The dynamical core.
    Q_0 : :class:`jax.numpy.Array`
        Wind forcing term in the vorticity equation.
    Q_network
        The right-hand-side forcing neural network.
    Q_callback : callable
        Passed `dynamics` and `Q_network`, and should return an additional term
        to be added to the right-hand-side of the vorticity equation. Evaluated
        before taking each timestep.
    N : Integral
        The number of timesteps to take using the dynamical solver between each
        output.
    n_output : Integral
        The number of outputs.
    input_weight : Real or :class:`jax.Array`
        Weight by which to scale each input.
    output_weight : Real or :class:`jax.Array`
        Weight by which to scale each output.
    """

    def __init__(self, dynamics, Q_0, Q_network, Q_callback, N, *, n_output=1,
                 input_weight=1, output_weight=1):
        super().__init__()
        self.__dynamics = dynamics
        self.__Q_0 = Q_0
        self.__Q_network = Q_network
        self.__Q_callback = Q_callback
        self.__N = N
        self.__n_output = n_output
        self.__input_weight = input_weight
        self.__output_weight = output_weight

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.__n_output) + input_shape[1:]

    def call(self, inputs):
        @jax.checkpoint
        def step(_, data):
            fields, dealias_fields, n = data
            dynamics = type(self.__dynamics)(self.__dynamics.parameters)
            # Workaround for missing GPU eig
            dynamics.poisson_solver = self.__dynamics.poisson_solver
            dynamics.modified_helmholtz_solver = self.__dynamics.modified_helmholtz_solver
            dynamics.fields.update(fields)
            dynamics.dealias_fields.update(dealias_fields)
            dynamics.n = n

            dynamics.fields["Q"] = self.__Q_0 + self.__Q_callback(dynamics, self.__Q_network)
            dynamics.step()

            data = (dict(dynamics.fields), dict(dynamics.dealias_fields), dynamics.n)
            return data

        def compute_step_outputs(data, _):
            data = jax.lax.fori_loop(0, self.__N, step, data, unroll=32)
            return data, data[0]["zeta"] * self.__output_weight

        def compute_outputs(zeta):
            dynamics = type(self.__dynamics)(self.__dynamics.parameters)
            # Workaround for missing GPU eig
            dynamics.poisson_solver = self.__dynamics.poisson_solver
            dynamics.modified_helmholtz_solver = self.__dynamics.modified_helmholtz_solver
            dynamics.initialize(zeta * self.__input_weight)
            data = (dict(dynamics.fields), dict(dynamics.dealias_fields), dynamics.n)
            _, outputs = jax.lax.scan(compute_step_outputs, data, None, length=self.__n_output)
            return outputs

        outputs = jax.vmap(compute_outputs)(inputs)
        return outputs


class OnlineDataset(keras.utils.PyDataset):
    """Online training data set.

    A time series is divided into sections, each with one input and a given
    number of outputs. The last output of one section is the input for a
    following section (if it exists). The index of the first section is
    randomized, and the sections themselves are randomized, with a new
    randomization on each epoch. Randomization uses
    :func:`numpy.random.randint` and :func:`numpy.random.random`.

    Parameters
    ----------

    h : :class:`zarr.hierarchy.Group`
        Parent group.
    input_path : str
        Input data group path.
    output_path : str
        Output data group path.
    n_output : Integral
        Number of outputs per input.
    batch_size : Integral
        Batch size.
    shuffle : bool
        Whether to apply randomization.
    input_weight
        Weight to apply to each input.
    output_weight
        Weight to apply to each output.
    i0 : Integral
        First index.
    i1 : Integral
        One past the last index. Defaults to the length of the data.
    """

    def __init__(self, h, input_path, output_path, n_output, batch_size, *,
                 shuffle=True, input_weight=1, output_weight=1, i0=0, i1=None):
        n = h[input_path].shape[-1]
        if h[output_path].shape[-1] != n:
            raise ValueError("Invalid length")
        if i1 is None:
            i1 = n
        if i1 - i0 - 1 - int(shuffle) * n_output <= 0:
            raise ValueError("Invalid length")
        if (i1 - i0 - 1 - int(shuffle) * n_output) % (n_output * batch_size) != 0:
            raise ValueError("Invalid length")

        super().__init__()
        self.__input_data = keras.ops.array(np.array(tuple(input_weight * h[input_path][..., i] for i in range(i0, i1))))
        if output_path == input_path and keras.ops.all(input_weight == output_weight):
            self.__output_data = self.__input_data
        else:
            self.__output_data = keras.ops.array(np.array(tuple(output_weight * h[output_path][..., i] for i in range(i0, i1))))
        self.__shuffle = shuffle
        self.__n_output = n_output
        self.__batch_size = batch_size

        self.__update_keys()

    def __update_keys(self):
        m = (self.__input_data.shape[0] - 1) // self.__n_output
        if self.__shuffle:
            m -= 1
            i0 = np.random.randint(1 + self.__n_output)
        else:
            i0 = 0
        assert m % self.__batch_size == 0

        input_keys = list(range(i0, i0 + m * self.__n_output, self.__n_output))
        assert len(input_keys) % self.__batch_size == 0
        output_keys = []
        for i in range(m):
            output_keys.append(tuple(range(i0 + 1 + i * self.__n_output,
                                           i0 + 1 + (i + 1) * self.__n_output)))

        p = [(i, o) for i, o in zip(input_keys, output_keys)]
        if self.__shuffle:
            np.random.shuffle(p)
        self.__input_keys = tuple(map(itemgetter(0), p))
        self.__output_keys = tuple(map(itemgetter(1), p))

    def __len__(self):
        assert len(self.__input_keys) % self.__batch_size == 0
        return len(self.__input_keys) // self.__batch_size

    def on_epoch_end(self):
        self.__update_keys()

    def __getitem__(self, key):
        if key < 0:
            key += len(self)
        if key < 0 or key >= len(self):
            raise KeyError("Invalid key")

        i0 = key * self.__batch_size
        i1 = (key + 1) * self.__batch_size
        inputs = keras.ops.reshape(
            self.__input_data[self.__input_keys[i0:i1], ...],
            (self.__batch_size,) + self.__input_data.shape[1:])
        outputs = keras.ops.reshape(
            self.__output_data[tuple(chain.from_iterable(self.__output_keys[i0:i1])), ...],
            (self.__batch_size, self.__n_output) + self.__output_data.shape[1:])

        return inputs, outputs
