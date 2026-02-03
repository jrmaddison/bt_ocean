"""Finite difference solvers for the 2D barotropic vorticity equation on a
beta-plane.
"""

try:
    import h5py
except ModuleNotFoundError:
    h5py = None
import jax
import jax.numpy as jnp
import keras
import numpy as np
try:
    import zarr
except ModuleNotFoundError:
    zarr = None

from abc import ABC, abstractmethod
from collections.abc import Mapping
from functools import cached_property, partial

from .fft import dst
from .grid import Grid
from .inversion import ModifiedHelmholtzSolver, PoissonSolver
from .precision import default_idtype, default_fdtype
from .pytree import PytreeNode

__all__ = \
    [
        "Parameters",
        "required",

        "Fields",

        "SteadyStateMaximumIterationsError",
        "NanEncounteredError",
        "Solver",
        "CNAB2Solver"
    ]

optional = object()
required = object()


class IOInterface:
    def __init__(self, h):
        self._h = h

    @property
    def h(self):
        return self._h

    @property
    def attrs(self):
        return self.h.attrs

    def create_group(self, name):
        return IOInterface(self.h.create_group(name))

    def create_dataset(self, name, shape=None, dtype=None, data=None):
        if shape is None and data is not None:
            shape = data.shape
        if dtype is None and data is not None:
            dtype = data.dtype

        if h5py is not None and isinstance(self.h, (h5py.File, h5py.Group)):
            self.h.create_dataset(name, shape=shape, dtype=dtype, data=data)
        elif zarr is not None and isinstance(self.h, zarr.Group):
            a = self.h.create_array(name, shape=shape, dtype=dtype)
            if data is not None:
                a[:] = data
        else:
            raise TypeError(f"Unexpected type: '{type(self.h)}'")


class Parameters(Mapping):
    """Model parameters.

    Parameters
    ----------

    parameters : Mapping
        Model parameters.
    defaults : Mapping
        Defines valid keys and default values. The sentinel value `required`
        indicates that a non-default value is required. The sentinel value
        `optional' can be used to indicate that the parameter is optional, but
        that no default value is required. If not supplied then all parameters
        are assumed to be optional.
    """

    def __init__(self, parameters, *, defaults=None):
        parameters = dict(parameters)
        if defaults is not None:
            for key, value in defaults.items():
                if key not in parameters and value is required:
                    raise KeyError(f"Missing parameter: '{key}'")
                if value is not optional:
                    parameters.setdefault(key, value)
            for key in parameters:
                if key not in defaults:
                    raise KeyError(f"Extra parameter: '{key}'")
        self._parameters = parameters

    def __getitem__(self, key):
        return self._parameters[key]

    def __iter__(self):
        yield from sorted(self._parameters)

    def __len__(self):
        return len(self._parameters)

    def write(self, h, path="parameters"):
        """Write parameters.

        Parameters
        ----------

        h : :class:`h5py.Group` or :class:`zarr.hierarchy.Group`
            Parent group.
        path : str
            Group path.

        Returns
        -------

        :class:`h5py.Group` or :class:`zarr.hierarchy.Group`
            Group storing the parameters.
        """

        h = IOInterface(h)
        g = h.create_group(path)
        g.attrs.update(self.items())
        return g.h

    @classmethod
    def read(cls, h, path="parameters"):
        """Read parameters.

        Parameters
        ----------

        h : :class:`h5py.Group` or :class:`zarr.hierarchy.Group`
            Parent group.
        path : str
            Group path.

        Returns
        -------

        :class:`.Parameters`
            The parameters.
        """

        return cls(h[path].attrs)


class Fields(Mapping):
    """Fields defined on a 2D grid.

    Fields values can be set using

    .. code::

        fields[key] = array

    and must be set before they can be accessed.

    Parameters
    ----------

    grid : :class:`.Grid`
        The 2D grid.
    keys : Iterable
        Field keys.
    """

    def __init__(self, grid, keys):
        keys = tuple(keys)
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate key")

        self._grid = grid
        self._keys = set(keys)
        self._fields = {}

    def __getitem__(self, key):
        if key not in self._keys:
            raise KeyError(f"Invalid key: '{key}'")
        if key not in self._fields:
            raise ValueError(f"Uninitialized value for key: '{key}'")
        return self._fields[key]

    def __setitem__(self, key, value):
        if key not in self._keys:
            raise KeyError(f"Invalid key: '{key}'")
        if value.shape != (self.grid.N_x + 1, self.grid.N_y + 1):
            raise ValueError(f"Invalid value for key: '{key}'")
        if value.dtype.type != self.grid.fdtype:
            raise ValueError(f"Invalid scalar data type for key: '{key}'")
        self._fields[key] = value

    def __iter__(self):
        yield from sorted(self._keys)

    def __len__(self):
        return len(self._keys)

    @property
    def grid(self) -> Grid:
        """The 2D grid.
        """

        return self._grid

    def zero(self, *keys):
        """Set fields equal to a zero-valued field.

        Parameters
        ----------

        keys : tuple
            The keys of fields to set equal to a zero-valued field.
        """

        for key in keys:
            self[key] = jnp.zeros((self.grid.N_x + 1, self.grid.N_y + 1),
                                  dtype=self.grid.fdtype)

    def clear(self, *, keep_keys=None):
        """Clear values for fields.

        Parameters
        ----------

        keep_keys : Iterable
            Keys for fields which should be retained.
        """

        if keep_keys is None:
            keep_keys = set()
        else:
            keep_keys = set(keep_keys)
        keep_keys = sorted(keep_keys)

        fields = {key: self[key] for key in keep_keys}
        self._fields.clear()
        self.update(fields)

    def write(self, h, path="fields"):
        """Write fields.

        Parameters
        ----------

        h : :class:`h5py.Group` or :class:`zarr.hierarchy.Group`
            Parent group.
        path : str
            Group path.

        Returns
        -------

        :class:`h5py.Group` or :class:`zarr.hierarchy.Group`
            Group storing the fields.
        """

        if not np.can_cast(self.grid.L_x, float):
            raise ValueError("Serialization not supported")
        if not np.can_cast(self.grid.L_y, float):
            raise ValueError("Serialization not supported")
        if not np.can_cast(self.grid.N_x, int):
            raise ValueError("Serialization not supported")
        if not np.can_cast(self.grid.N_y, int):
            raise ValueError("Serialization not supported")

        h = IOInterface(h)
        g = h.create_group(path)
        del h
        g.attrs["L_x"] = float(self.grid.L_x)
        g.attrs["L_y"] = float(self.grid.L_y)
        g.attrs["N_x"] = int(self.grid.N_x)
        g.attrs["N_y"] = int(self.grid.N_y)
        g.attrs["idtype"] = jnp.dtype(self.grid.idtype).name
        g.attrs["fdtype"] = jnp.dtype(self.grid.fdtype).name

        for key, value in self.items():
            g.create_dataset(
                name=key, data=np.array(value, dtype=self.grid.fdtype))

        return g.h

    @classmethod
    def read(cls, h, path="fields", *, grid=None):
        """Read fields.

        Parameters
        ----------

        h : :class:`h5py.Group` or :class:`zarr.hierarchy.Group`
            Parent group.
        path : str
            Group path.
        grid : :class:`.Grid`
            The 2D grid.

        Returns
        -------

        :class:`.Fields`
            The fields.
        """

        g = h[path]
        del h

        L_x = g.attrs["L_x"]
        L_y = g.attrs["L_y"]
        N_x = g.attrs["N_x"]
        N_y = g.attrs["N_y"]
        idtype = jnp.dtype(g.attrs["idtype"]).type
        fdtype = jnp.dtype(g.attrs["fdtype"]).type
        if grid is None:
            grid = Grid(L_x, L_y, N_x, N_y, idtype=idtype, fdtype=fdtype)
        if L_x != grid.L_x or L_y != grid.L_y:
            raise ValueError("Invalid dimension(s)")
        if N_x != grid.N_x or N_y != grid.N_y:
            raise ValueError("Invalid number(s) of divisions")
        if idtype != grid.idtype or fdtype != grid.fdtype:
            raise ValueError("Invalid dtype(s)")

        fields = cls(grid, set(g))
        for key in g:
            fields[key] = g[key][...]

        return fields

    def update(self, d):
        """Update field values from the supplied :class:`Mapping`.

        Parameters
        ----------

        d : Mapping
            Key-value pairs containing the field values.
        """

        for key, value in d.items():
            self[key] = value


class SteadyStateMaximumIterationsError(Exception):
    """Raised when a steady-state solve exceeds the maximum permitted number of
    iterations.
    """


class NanEncounteredError(Exception):
    """Raised when a NaN is encountered.
    """


class Solver(PytreeNode, ABC):
    r"""Finite difference solver for the 2D barotropic vorticity equation on a
    beta-plane,

    .. math::

        \partial_t \zeta
            + \nabla \cdot ( (\nabla^\perp \psi) \zeta )
            + \beta \partial_x \psi
            = \nu \nabla^2 \zeta
            - r \zeta
            + Q,

    where

    .. math::

        \nabla^2 \psi = \zeta,

    subject to no-normal flow and free-slip boundary conditions.

    Parameters
    ----------

    parameters : :class:`.Parameters`
        Model parameters.

            - `'L_x'` : Defines the :math:`x`-dimension extents, :math:`x \in
              [ -L_x, L_x ]`.
            - `'L_y'` : Defines the :math:`y`-dimension extents, :math:`y \in
              [ -L_y, L_y ]`.
            - `'N_x'` : Number of :math:`x`-dimension divisions.
            - `'N_y'` : Number of :math:`y`-dimension divisions.
            - `'\beta'` : :math:`y` derivative of the Coriolis parameter,
              :math:`\beta`.
            - `'\nu'` : Laplacian viscosity coefficient, :math:`\nu`.
            - `'r'` : Linear drag coefficient, :math:`r`.
            - `dt` : Time step size.

    idtype : type
        Integer scalar data type. Defaults to :func:`.default_idtype()`.
    fdtype : type
        Floating point scalar data type. Defaults to :func:`.default_fdtype()`.
    field_keys : Iterable
        Keys for fields. The following keys are added by default

            - `'psi'` : The current stream function field.
            - `'zeta'` : The current relative vorticity field.
            - `'Q'` : A field defining an extra term in the vorticity equation,
              :math:`Q`. Defaults to a zero-valued field.

    prescribed_field_keys : Iterable
        Keys for fields which are prescribed, and which are not updated within
        a timestep. Defaults to `{'Q'}`.
    """

    _registry = {}

    _defaults = {"L_x": required,  # x \in [-L_x, L_x]
                 "L_y": required,  # y \in [-L_y, L_y]
                 "N_x": required,
                 "N_y": required,
                 "beta": required,
                 "r": required,
                 "nu": required,
                 "dt": required}

    def __init__(self, parameters, *, idtype=None, fdtype=None,
                 field_keys=None, prescribed_field_keys=None):
        self._parameters = parameters = Parameters(parameters, defaults=self._defaults)
        if idtype is None:
            idtype = default_idtype()
        if fdtype is None:
            fdtype = default_fdtype()
        idtype = jnp.dtype(idtype).type
        fdtype = jnp.dtype(fdtype).type
        if field_keys is None:
            field_keys = set()
        else:
            field_keys = set(field_keys)
        field_keys.update({"psi", "zeta", "Q"})
        if prescribed_field_keys is None:
            prescribed_field_keys = {"Q"}

        self._grid = grid = Grid(
            parameters["L_x"], parameters["L_y"],
            parameters["N_x"], parameters["N_y"],
            idtype=idtype, fdtype=fdtype)

        self._fields = Fields(grid, field_keys)
        self._prescribed_field_keys = tuple(sorted(prescribed_field_keys))

        self.zero_prescribed()
        self.initialize()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        cls._registry[cls.__name__] = cls
        keras.saving.register_keras_serializable(package=f"_bt_ocean__{cls.__name__}")(cls)

    @cached_property
    def beta(self) -> jax.Array:
        r""":math:`y` derivative of the Coriolis parameter, :math:`\beta`.
        """

        return jnp.array(self.parameters["beta"], dtype=self.grid.fdtype)

    @cached_property
    def nu(self) -> jax.Array:
        r"""Laplacian viscosity coefficient, :math:`\nu`.
        """

        return jnp.array(self.parameters["nu"], dtype=self.grid.fdtype)

    @cached_property
    def r(self) -> jax.Array:
        """Linear drag coefficient, :math:`r`.
        """

        return jnp.array(self.parameters["r"], dtype=self.grid.fdtype)

    @cached_property
    def dt(self) -> jax.Array:
        """Time step size.
        """

        return jnp.array(self.parameters["dt"], dtype=self.grid.fdtype)

    @property
    def n(self):
        """The number of timesteps which have been taken.
        """

        return self._n

    @n.setter
    def n(self, n):
        self._n = n

    @property
    def grid(self) -> Grid:
        """2D grid.
        """

        return self._grid

    @property
    def parameters(self) -> Parameters:
        """Model parameters.
        """

        return self._parameters

    @property
    def fields(self) -> Fields:
        """Fields.
        """

        return self._fields

    @property
    def prescribed_field_keys(self) -> tuple:
        """Keys of prescribed fields.
        """

        return self._prescribed_field_keys

    @cached_property
    def poisson_solver(self) -> PoissonSolver:
        """Solver for the Poisson equation.
        """

        return PoissonSolver(self.grid)

    def zero_prescribed(self):
        """Zero prescribed fields.
        """

        self.fields.zero(*self.prescribed_field_keys)

    @abstractmethod
    def initialize(self, zeta=None):
        """Initialize the model.

        Parameters
        ----------

        zeta : :class:`jax.Array`
            Initial relative vorticity field. `None` indicates a zero-valued
            field.
        """

        self.fields.clear(keep_keys=self.prescribed_field_keys)
        self._n = 0

    @abstractmethod
    def step(self):
        """Take a timestep.
        """

        self._n += 1

    @staticmethod
    @jax.jit
    def _step(_, model):
        model = model.copy()
        model.step()
        return model

    def steps(self, n, *, unroll=8):
        """Take multiple timesteps. Uses :func:`jax.lax.fori_loop`.

        Parameters
        ----------

        n : Integral
            The number of timesteps to take.
        unroll : Integral
            Passed to :func:`jax.lax.fori_loop`.
        """

        model = jax.lax.fori_loop(0, n, self._step, self, unroll=unroll)
        self.update(model)

    def ke(self):
        """The current kinetic energy, divided by density.

        Returns
        -------

        Real
            The current kinetic energy.
        """

        u = -self.grid.D_y(self.fields["psi"])
        v = self.grid.D_x(self.fields["psi"])
        return 0.5 * self.grid.integrate(u * u + v * v)

    def ke_spectrum(self):
        """The current 2D kinetic energy spectrum.

        Computed using a type-I DST. The spectrum is defined so that the sum of
        the spectrum equals the current kinetic energy divided by the density.

        Returns
        -------

        :class:`jax.Array`
            The current 2D kinetic energy spectrum.
        """

        psi = self.fields["psi"]
        I = jnp.outer(jnp.arange(self.grid.N_x + 1, dtype=self.grid.idtype),
                      jnp.ones(self.grid.N_y + 1, dtype=self.grid.idtype))
        J = jnp.outer(jnp.ones(self.grid.N_x + 1, dtype=self.grid.idtype),
                      jnp.arange(self.grid.N_y + 1, dtype=self.grid.idtype))
        k = jnp.pi * I / (2 * self.grid.L_x)
        l = jnp.pi * J / (2 * self.grid.L_y)
        return (0.5 * self.grid.L_x * self.grid.L_y
                * (k ** 2 + l ** 2) * (dst(dst(psi, axis=0), axis=1) ** 2))

    def steady_state_solve(self, *args, update=lambda model, *args: None, tol, max_it=10000, _min_n=0):
        r"""Timestep to steady-state.

        Uses timestepping to define a fixed-point iteration, and applies
        reverse mode differentiation using a two-phase approach.

        References:

            - Andreas Griewank and Andrea Walther, 'Evaluating derivatives',
              second edition, Society for Industrial and Applied Mathematics,
              2008, ISBN: 978-0-898716-59-7, chapter 15
            - Bruce Christianson, 'Reverse accumulation and attractive fixed
              points', Optimization Methods and Software 3(4), pp. 311--326
              1994, doi: https://doi.org/10.1080/10556789408805572
            - Zico Kolter, David Duvenaud, and Matt Johnson, 'Deep implicit
              layers - neural ODEs, deep equilibirum models, and beyond',
              https://implicit-layers-tutorial.org/ [accessed 2024-08-26],
              chapter 2

        Parameters
        ----------

        update : callable
            A callable accepting a :class:`.Solver` as the zeroth argument and
            the elements of `m` as remaining positional arguments, and which
            updates the values of control variables.
        args : tuple
            Passed to `update`.
        tol : Real
            Tolerance. The system is timestepped until

            .. math::

                \left\| \zeta_{n + 1} - \zeta_n \right\|_\infty
                    \le \varepsilon \left\| \zeta_{n + 1} \right\|_\infty,

            where :math:`\zeta_n` is the degree-of-freedom vector on timestep
            :math:`n` and :math:`\varepsilon` is the tolerance.
        max_it : Integral
            Maximum number of iterations.

        Returns
        -------

        int
            The number of iterations.
        """

        @jax.jit
        def forward_step(data, args):
            _, model, it = data
            model = model.copy()
            zeta_0 = model.fields["zeta"]
            update(model, *args)
            model.step()
            return (zeta_0, model, it + 1)

        @jax.custom_vjp
        def forward(model, args):
            while model.n < _min_n:
                zeta_0, model, _ = forward_step((None, model, 0), args)
            zeta_0, model, _ = forward_step((None, model, 0), args)

            def non_convergence(data):
                zeta_0, model, it = data
                zeta_1 = model.fields["zeta"]
                return jnp.logical_and(
                    it <= max_it,
                    abs(zeta_1 - zeta_0).max() > tol * abs(zeta_1).max())

            _, model, it = jax.lax.while_loop(
                non_convergence, partial(forward_step, args=args),
                (zeta_0, model, 1))
            return model, it

        def forward_fwd(model, args):
            model, it = forward(model, args)
            return (model, it), (model, args)

        def forward_bwd(res, zeta):
            model, args = res
            zeta_model, _ = zeta

            _, vjp_step = jax.vjp(
                lambda model: forward_step((None, model, 0), args)[1], model)

            @jax.jit
            def adj_step(data, zeta_model):
                _, lam_model, lam_it = data
                lam_zeta_0 = lam_model.fields["zeta"]
                lam_model.zero_prescribed()
                lam_model, = vjp_step(lam_model)
                for key, value in lam_model.fields.items():
                    lam_model.fields[key] = value + zeta_model.fields[key]
                return lam_zeta_0, lam_model, lam_it + 1

            def adjoint(zeta_model):
                lam_model = zeta_model
                lam_zeta_0, lam_model, _ = adj_step((None, lam_model, 0), zeta_model)

                def non_convergence(data):
                    lam_zeta_0, lam_model, lam_it = data
                    lam_zeta_1 = lam_model.fields["zeta"]
                    return jnp.logical_and(
                        lam_it <= max_it,
                        # l^1 norm (dual to l^\infty)
                        abs(lam_zeta_1 - lam_zeta_0).sum() > tol * abs(lam_zeta_1).sum())

                _, lam_model, lam_it = jax.lax.while_loop(
                    non_convergence, partial(adj_step, zeta_model=zeta_model), (lam_zeta_0, lam_model, 1))
                if lam_it > max_it:
                    raise SteadyStateMaximumIterationsError("Maximum number of iterations exceeded")
                return lam_model

            lam_model = adjoint(zeta_model)
            _, vjp = jax.vjp(lambda args: forward_step((None, model, 0), args)[1], args)
            lam_args, = vjp(lam_model)

            return lam_model, lam_args

        forward.defvjp(forward_fwd, forward_bwd)

        model, it = forward(self, args)
        self.update(model)
        if it > max_it:
            raise SteadyStateMaximumIterationsError("Maximum number of iterations exceeded")

        return it

    def write(self, h, path="solver"):
        """Write solver.

        Parameters
        ----------

        h : :class:`h5py.Group` or :class:`zarr.hierarchy.Group`
            Parent group.
        path : str
            Group path.

        Returns
        -------

        :class:`h5py.Group` or :class:`zarr.hierarchy.Group`
            Group storing the solver.
        """

        h = IOInterface(h)
        g = h.create_group(path)
        del h

        g.attrs["type"] = type(self).__name__
        g.attrs["n"] = int(self.n)
        self.parameters.write(g.h, "parameters")
        self.fields.write(g.h, "fields")

        return g.h

    @classmethod
    def read(cls, h, path="solver"):
        """Read solver.

        Parameters
        ----------

        h : :class:`h5py.Group` or :class:`zarr.hierarchy.Group`
            Parent group.
        path : str
            Group path.

        Returns
        -------

        :class:`.Solver`
            The solver.
        """

        g = h[path]
        del h

        cls = cls._registry[g.attrs["type"]]
        idtype = jnp.dtype(g["fields"].attrs["idtype"]).type
        fdtype = jnp.dtype(g["fields"].attrs["fdtype"]).type
        model = cls(Parameters.read(g, "parameters"), idtype=idtype, fdtype=fdtype)
        model.fields.update(Fields.read(g, "fields", grid=model.grid))
        model.n = g.attrs["n"]

        return model

    def new(self, *, copy_prescribed=False):
        """Return a new :class:`.Solver` with the same configuration as this
        :class:`.Solver`.

        Parameters
        ----------

        copy_prescribed : bool
            Whether to copy values of prescribed fields to the new
            :class:`.Solver`.

        Returns
        -------

        :class:`.Solver`
            The new :class:`.Solver`.
        """

        model = type(self)(self.parameters, idtype=self.grid.idtype, fdtype=self.grid.fdtype)
        if copy_prescribed:
            for key in self.prescribed_field_keys:
                model.fields[key] = self.fields[key]
        return model

    def update(self, model):
        """Update the state of this :class:`.Solver`.

        Parameters
        ----------

        model : :class:`.Solver`
            Defines the new state of this :class:`.Solver`.
        """

        self.fields.update(model.fields)
        self.n = model.n

    def copy(self):
        """Return a copy of the model.

        Returns
        -------

        :class:`.Solver`
            A copy of the model.
        """

        model = self.new()
        model.update(self)
        return model

    def flatten(self):
        """Return a JAX flattened representation.

        Returns
        -------

        Sequence[Sequence[object, ...], Sequence[object, ...]]
        """

        return ((dict(self.fields), self.n),
                (self.parameters, self.grid.idtype, self.grid.fdtype))

    @classmethod
    def unflatten(cls, aux_data, children):
        """Unpack a JAX flattened representation.
        """

        parameters, idtype, fdtype = aux_data
        fields, n = children

        model = cls(parameters, idtype=idtype, fdtype=fdtype)
        model.fields.update({key: value for key, value in fields.items() if type(value) is not object})
        if type(n) is not object:
            model.n = n

        return model

    def get_config(self):
        return {"type": type(self).__name__,
                "parameters": dict(self.parameters),
                "idtype": jnp.dtype(self.grid.idtype).name,
                "fdtype": jnp.dtype(self.grid.fdtype).name,
                "fields": dict(self.fields),
                "n": self.n}

    @classmethod
    def from_config(cls, config):
        config = {key: keras.saving.deserialize_keras_object(value) for key, value in config.items()}
        cls = cls._registry[config["type"]]
        if "fdtype" in config:
            fdtype = jnp.dtype(config["fdtype"]).type
        else:
            # Backwards compatibility
            fdtype = config["fields"]["Q"].dtype.type
        if "idtype" in config:
            idtype = jnp.dtype(config["idtype"]).type
        else:
            # Backwards compatibility
            idtype = jnp.dtype({"float32": jnp.int32, "float64": jnp.int64}[jnp.dtype(fdtype).name]).type
        model = cls(config["parameters"], idtype=idtype, fdtype=fdtype)
        model.fields.update(config["fields"])
        model.n = config["n"]
        return model


class CNAB2Solver(Solver):
    """Finite difference solver for the 2D barotropic vorticity equation on a
    beta-plane, using a CNAB2 time discretization.

    CNAB2 reference:

        - Uri M. Ascher, Steven J. Ruuth, and Brian T. R. Wetton,
          'Implicit-explicit methods for time-dependent partial differential
          equations', SIAM Journal on Numerical Analysis 32(3), 797--823, 1995,
          https://doi.org/10.1137/0732037

    Parameters
    ----------

    See :class:`.Solver`.
    """

    def __init__(self, parameters, *, idtype=None, fdtype=None):
        super().__init__(
            parameters, idtype=idtype, fdtype=fdtype,
            field_keys={"F_1"})

    @cached_property
    def modified_helmholtz_solver(self) -> ModifiedHelmholtzSolver:
        """Modified Helmholtz solver used for the implicit time discretization.
        """

        return ModifiedHelmholtzSolver(
            self.grid, alpha=1 + 0.5 * self.dt * self.r, beta=0.5 * self.dt * self.nu)

    def initialize(self, zeta=None):
        super().initialize(zeta=zeta)
        if zeta is None:
            self.fields.zero("psi", "zeta")
        else:
            self._update_fields(zeta)
        self.fields.zero("F_1")

    def _update_fields(self, zeta):
        zeta_0 = jnp.zeros_like(zeta)
        zeta_0 = zeta_0.at[1:-1, 1:-1].set(zeta[1:-1, 1:-1])
        zeta = zeta_0
        del zeta_0

        if not isinstance(zeta, jax._src.core.Tracer) and jnp.isnan(zeta).any():
            raise NanEncounteredError("nan encountered")
        psi = self.poisson_solver.solve(zeta)
        if not isinstance(psi, jax._src.core.Tracer) and jnp.isnan(psi).any():
            raise NanEncounteredError("nan encountered")

        self.fields["psi"] = psi
        self.fields["zeta"] = zeta

    def step(self):
        psi = self.fields["psi"]
        zeta = self.fields["zeta"]
        q = zeta + self.beta * self.grid.Y

        F_0 = (
            self.grid.J(q, psi)
            + self.fields["Q"])
        G_0 = (
            self.nu * (self.grid.D_xx(zeta, boundary=False) + self.grid.D_yy(zeta, boundary=False))
            - self.r * zeta)

        coeff_0, coeff_1 = jax.lax.select(
            jnp.full((2,), self.n == 0, dtype=bool),
            jnp.array((1.0, 0.0), dtype=self.grid.fdtype),
            jnp.array((1.5, -0.5), dtype=self.grid.fdtype))
        F = coeff_0 * F_0 + coeff_1 * self.fields["F_1"]

        b = jnp.zeros_like(zeta).at[1:-1, 1:-1].set(
            (zeta + self.dt * (F + 0.5 * G_0))[1:-1, 1:-1])
        zeta = self.modified_helmholtz_solver.solve(-b)

        self._update_fields(zeta)
        self.fields["F_1"] = F_0
        super().step()

    def steady_state_solve(self, *args, update=lambda model, *args: None, tol, max_it=10000):
        return super().steady_state_solve(*args, update=update, tol=tol, max_it=max_it, _min_n=1)
