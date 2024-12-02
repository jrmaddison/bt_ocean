"""Diagnostics.
"""

import jax.numpy as jnp
import numpy as np

from abc import ABC, abstractmethod
import csv
import itertools
from numbers import Real
import operator
import scipy

from .grid import Grid
from .model import Fields
from .precision import x64_enabled

__all__ = \
    [
        "AverageDefinition",
        "FieldAverage",
        "FieldProductAverage",

        "Average",

        "Diagnostic",
        "Timestep",
        "Time",
        "KineticEnergy",
        "SeparationPoint",
        "DiagnosticsCsv"
    ]


class AverageDefinition:
    """Defines a quantity to be averaged.

    Parameters
    ----------

    key : str
        The key which will be used to store the average.
    field_keys : Sequence[str, ...]
        :class:`.Fields` keys which define arguments to `op`.
    op : callable
        A callable accepting zero or more :class:`jnp.Array` positional
        arguments and returning a :class:`jnp.Array` defining the quantity to
        be averaged.

    Notes
    -----

    `op` arguments are defined by `(fields[key] for key in field_keys)`. The
    arrays passed to `op`, and the result, are cast to double precision.
    """

    def __init__(self, key, field_keys, op):
        self._key = key
        self._field_keys = tuple(field_keys)
        self._op = op

    @property
    def key(self) -> str:
        """The key which will be used to store the average.
        """

        return self._key

    @x64_enabled()
    def evaluate(self, fields):
        """Compute a value of the quantity to be averaged.

        Parameters
        ----------

        fields : :class:`.Fields`
            Input data.

        Returns
        -------

        :class:`jax.Array`
            The value of the quantity to be averaged.
        """

        args = tuple(jnp.array(fields[key], dtype=jnp.float64)
                     for key in self._field_keys)
        return jnp.array(self._op(*args), dtype=jnp.float64)


class FieldAverage(AverageDefinition):
    """Defines a field to be averaged.

    Parameters
    ----------

    field_key : str
        :class:`.Fields` key defining the field to be averaged.
    key : str
        The key which will be used to store the average. Defaults to
        `field_key`.
    """

    def __init__(self, field_key, *, key=None):
        if key is None:
            key = field_key
        super().__init__(key, (field_key,), op=lambda x: x)


class FieldProductAverage(AverageDefinition):
    """Defines a product to be averaged.

    Parameters
    ----------

    field_key_a : str
        :class:`.Fields` key defining the first argument of the multiply to be
        averaged.
    field_key_b : str
        :class:`.Fields` key defining the second argument of the multiply to be
        averaged.
    key : str
        The key which will be used to store the average. Defaults to
        `f"{field_key_a}_{field_key_b}"`.
    """

    def __init__(self, field_key_a, field_key_b, *, key=None):
        if key is None:
            key = f"{field_key_a}_{field_key_b}"
        super().__init__(key, (field_key_a, field_key_b), op=operator.mul)


def zarr_append(d, value, *, axis=None):
    """Append data to a :class:`zarr.core.Array`.

    Parameters
    ----------

    d : :class:`zarr.core.Array`
        The :class:`zarr.core.Array` to append to.
    value : Sequence
        The data to append.
    axis : Integral
        The append axis. Defaults to `len(value.shape)`.
    """

    value = np.asarray(value)
    if axis is None:
        axis = len(value.shape)
    shape = list(value.shape)
    shape.insert(axis, 1)
    shape = tuple(shape)

    value = np.reshape(value, shape)
    d.append(value, axis=axis)


class Average:
    """Averaging utility class.

    Parameters
    ----------

    grid : :class:`.Grid`
        Defines the grid on which to store averaged fields. Note that a new
        grid with double precision floating point scalar data type is used
        internally.
    definitions : Sequence[:class:`.AverageDefinition`, ...]
        Defines the quantities to be averaged.

    Notes
    -----

    All averaging calculations are performed in double precision.
    """

    _reserved_keys = {"w"}

    @x64_enabled()
    def __init__(self, grid, definitions):
        if len(set(definition.key for definition in definitions).intersection(
                self._reserved_keys)) > 0:
            raise ValueError("Reserved key(s)")

        grid = Grid(grid.L_x, grid.L_y, grid.N_x, grid.N_y,
                    idtype=jnp.int64, fdtype=jnp.float64)

        self._definitions = tuple(definitions)
        self._fields = Fields(
            grid, tuple(definition.key for definition in definitions))

        self.zero()

    @property
    def grid(self) -> Grid:
        """Grid.
        """

        return self._fields.grid

    def keys(self):
        """Averaged data keys.

        Returns
        -------

        Iterable[str, ...]
            The keys.
        """

        return self._fields.keys()

    @property
    def w(self) -> Real:
        """The current sum of weights.
        """

        return self._w

    @x64_enabled()
    def zero(self):
        """Reset and zero all fields.
        """

        self._fields.zero(*self._fields)
        self._w = jnp.float64(0)

    @x64_enabled()
    def add(self, fields, *, weight=1):
        """Add to averaged fields.

        Parameters
        ----------

        fields : :class:`.Fields`
            Input data.
        weight : Real
            Multiplication weight.
        """

        weight = jnp.float64(weight)
        for definition in self._definitions:
            self._fields[definition.key] = self._fields[definition.key] + weight * definition.evaluate(fields)
        self._w += weight

    @x64_enabled()
    def averaged_fields(self):
        """Return averaged fields.

        Returns
        -------

        :class:`.Fields`
            The averaged fields.

        Notes
        -----

        Computes the average by dividing summed quantities by the sum of all
        weights.
        """

        if self.w == 0:
            raise RuntimeError("Zero weight")

        fields = Fields(self.grid, tuple(self._fields))
        for key, field in self._fields.items():
            fields[key] = field / self.w
        return fields

    @x64_enabled()
    def append_averaged_fields(self, h, path=""):
        """Append averaged field data to a :class:`zarr.hierarchy.Group`.

        Parameters
        ----------

        h : :class:`zarr.hierarchy.Group`
            Parent group.
        path : str
            Group path.

        Returns
        -------

        :class:`zarr.hierarchy.Group`
            Group storing the data.
        """

        if path not in h:
            h.create_group(path)
        g = h[path]
        del h

        if "w" not in g:
            g.create_dataset(
                "w", shape=(1, 0), dtype=self.w.dtype)
        for key in self.keys():
            if key not in g:
                g.create_dataset(
                    key, shape=(self.grid.N_x + 1, self.grid.N_y + 1, 0),
                    dtype=self.grid.fdtype, chunks=(-1, -1, 1))

        zarr_append(g["w"], (self.w,))
        for key, value in self.averaged_fields().items():
            zarr_append(g[key], value)

        return g


def zero_point(x, y, i):
    """Defining :math:`y(x)` via linear interpolation

    .. math::

        (y(x) - y_0) (x_1 - x_0) = (y_1 - y_0) (x - x_0),

    return the value of :math:`x` for which :math:`y(x) = 0`. Requires
    :math:`y_0` and :math:`y_1` to be non-equal and of differing sign.

    Parameters
    ----------

    x : :class:`np.ndarray` or :class:`jax.Array`
        `x[i]` and `x[i + 1]` define :math:`x_0` and :math:`x_1` respectively.
    y : :class:`np.ndarray` or :class:`jax.Array`
        `y[i]` and `y[i + 1]` define :math:`y_0` and :math:`y_1` respectively.
    i : Integral
        Index.

    Returns
    -------

    Real
        The value of :math:`x` for which :math:`y(x) = 0`.
    """

    if y[i + 1] == y[i]:
        raise ValueError("Divide by zero")
    if y[i] * y[i + 1] > 0:
        raise ValueError("Sign definite")
    xz = x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
    if xz < min(x[i], x[i + 1]):
        raise ValueError("Out of bounds")
    if xz > max(x[i], x[i + 1]):
        raise ValueError("Out of bounds")
    return xz


class Diagnostic(ABC):
    """Defines one or more diagnostics.
    """

    @property
    @abstractmethod
    def names(self) -> tuple[str, ...]:
        """Diagnostic names.
        """

        raise NotImplementedError

    @abstractmethod
    def values(self, model):
        """Compute diagnostic values.

        Parameters
        ----------

        model : :class:`.Solver`
            Model.

        Returns
        -------

        tuple[object, ...]
            Diagnostic values.
        """

        raise NotImplementedError


class Timestep(Diagnostic):
    """Computes the following diagnostic:

        - `'n'` : The model timestep number.
    """

    @property
    def names(self):
        return ("n",)

    def values(self, model):
        return (model.n,)


class Time(Diagnostic):
    """Computes the following diagnostic:

        - `'t'` : The model time.
    """

    @property
    def names(self):
        return ("t",)

    def values(self, model):
        return (model.n * model.dt,)


class KineticEnergy(Diagnostic):
    """Computes the following diagnostic:

        - `'ke'` : The kinetic energy, divided by density.
    """

    @property
    def names(self):
        return ("ke",)

    def values(self, model):
        return (model.ke(),)


class SeparationPoint(Diagnostic):
    """Computes the following diagnostic:

        - `'y_sep'` : The jet separation :math:`y`-coordinate, computed as the
          the location in the central half of the domain at which the northward
          component of the velocity, on the western boundary, changes sign.

    `'y_sep'` is computed via linear interpolation between grid point values.
    """

    @property
    def names(self):
        return ("y_sep",)

    def values(self, model):
        grid = model.grid
        v = model.grid.D_x(model.fields["psi"])

        v = v[0, :]
        j0 = grid.N_y // 4
        j1 = grid.N_y - j0
        if ((v[j0 + 1:j1] * v[j0:j1 - 1]) < 0).sum() != 1:
            return (jnp.nan,)
        j0 = j0 + jnp.argmin(v[j0 + 1:j1] * v[j0:j1 - 1])
        j1 = j0 + 1
        assert v[j0] * v[j1] < 0

        return (zero_point(grid.y, v, j0),)


class DiagnosticsCsv:
    """Output diagnostics csv file.

    Parameters
    ----------

    h
        File handle.
    diagnostics : tuple[Diagnostic, ...]
        Diagnostics.
    args
        Passed to :func:`csv.writer`.
    kwargs
        Passed to :func:`csv.writer`.
    """

    def __init__(self, h, diagnostics, *args, **kwargs):
        diagnostics = tuple(diagnostics)
        names = tuple(itertools.chain.from_iterable(diag.names for diag in diagnostics))
        if len(set(names)) != len(names):
            raise ValueError("Duplicate names")

        self._h = h
        self._csv = csv.writer(h, *args, **kwargs)
        self._diagnostics = diagnostics
        self._csv.writerow(names)
        self._h.flush()

    def write(self, model):
        """Write diagnostics.

        Parameters
        ----------

        model : :class:`.Solver`
            Model.
        """

        values = tuple(diag.values(model) for diag in self._diagnostics)
        if tuple(len(diag.names) for diag in self._diagnostics) != tuple(len(value) for value in values):
            raise RuntimeError("Unexpected length")
        values = tuple(itertools.chain.from_iterable(values))
        self._csv.writerow(tuple(map(lambda value: f"{value}", values)))
        self._h.flush()
