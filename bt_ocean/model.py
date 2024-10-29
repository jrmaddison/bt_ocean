"""Chebyshev pseudospectral solvers for the barotropic vorticity equation on a
beta plane.
"""

import jax
import jax.numpy as jnp
import keras
import numpy as np

from abc import ABC, abstractmethod
from collections.abc import Mapping
from functools import cached_property, partial

from .fft import dst
from .grid import Grid, SpectralGridTransfer
from .inversion import ModifiedHelmholtzSolver, PoissonSolver

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


class Required:
    pass


required = Required()


class Parameters(Mapping):
    """Model parameters.

    Parameters
    ----------

    parameters : Mapping
        Model parameters.
    defaults : Mapping
        Defines default values. The sentinel value `required` indicates that
        a non-default value is required.
    """

    def __init__(self, parameters, *, defaults=None):
        if defaults is None:
            defaults = {}

        parameters = dict(parameters)
        for key, value in defaults.items():
            if key not in parameters and value is required:
                raise KeyError(f"Missing parameter: '{key}'")
            parameters.setdefault(key, value)
        self._parameters = parameters

    def __getitem__(self, key):
        return self._parameters[key]

    def __iter__(self):
        yield from sorted(self._parameters)

    def __len__(self):
        return len(self._parameters)

    def write(self, h, path="parameters"):
        """Write parameters to a :class:`zarr.hierarchy.Group`.

        Parameters
        ----------

        h : :class:`zarr.hierarchy.Group`
            Parent group.
        path : str
            Group path.

        Returns
        -------

        :class:`zarr.hierarchy.Group`
            Group storing the parameters.
        """

        g = h.create_group(path)
        g.attrs.update(self.items())
        return g

    @classmethod
    def read(cls, h, path="parameters"):
        """Read parameters from a :class:`zarr.hierarchy.Group`.

        Parameters
        ----------

        h : :class:`zarr.hierarchy.Group`
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
    """Fields defined on a 2D Chebyshev grid.

    Fields values can be set using

    .. code::

        fields[key] = array

    and must be set before they can be accessed.

    Parameters
    ----------

    grid : :class:`.Grid`
        The 2D Chebyshev grid.
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
        """The 2D Chebyshev grid.
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
        """Write fields to a :class:`zarr.hierarchy.Group`.

        Parameters
        ----------

        h : :class:`zarr.hierarchy.Group`
            Parent group.
        path : str
            Group path.

        Returns
        -------

        :class:`zarr.hierarchy.Group`
            Group storing the fields.
        """

        g = h.create_group(path)
        del h

        g.attrs["L_x"] = self.grid.L_x
        g.attrs["L_y"] = self.grid.L_y
        g.attrs["N_x"] = self.grid.N_x
        g.attrs["N_y"] = self.grid.N_y
        g.attrs["idtype"] = jnp.dtype(self.grid.idtype).name
        g.attrs["fdtype"] = jnp.dtype(self.grid.fdtype).name

        for key in self:
            g.create_dataset(
                name=key, data=np.array(self[key], dtype=self.grid.fdtype))

        return g

    def update(self, d):
        """Update field values from the supplied :class:`Mapping`.

        Parameters
        ----------

        d : Mapping
            Key-value pairs containing the field values.
        """

        for key, value in d.items():
            self[key] = value

    @classmethod
    def read(cls, h, path="fields", *, grid=None):
        """Read fields from a :class:`zarr.hierarchy.Group`.

        Parameters
        ----------

        h : :class:`zarr.hierarchy.Group`
            Parent group.
        path : str
            Group path.
        grid : :class:`.Grid`
            The 2D Chebyshev grid.

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
            raise ValueError("Invalid degree(s)")
        if idtype != grid.idtype or fdtype != grid.fdtype:
            raise ValueError("Invalid dtype(s)")

        fields = cls(grid, set(g))
        for key in g:
            fields[key] = g[key][...]

        return fields


class SteadyStateMaximumIterationsError(Exception):
    """Raised when a steady-state solve exceeds the maximum permitted number of
    iterations.
    """


class NanEncounteredError(Exception):
    """Raised when a NaN is encountered.
    """


class Solver(ABC):
    _registry = {}

    r"""Chebyshev pseudospectral solver for the 2D barotropic vorticity
    equation on a beta-plane,

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
            - `'N_x'` : :math:`x`-dimension Chebyshev degree.
            - `'N_y'` : :math:`y`-dimension Chebyshev degree.
            - `'\beta'` : :math:`y` derivative of the Coriolis parameter,
              :math:`\beta`.
            - `'\nu'` : Laplacian viscosity coefficient, :math:`\nu`.
            - `'r'` : Linear drag coefficient, :math:`r`.
            - `dt` : Time step size.

    field_keys : Iterable
        Keys for fields defined on the 'base' grid. The following keys are
        added by default

            - `'psi'` : The current stream function field.
            - `'zeta'` : The current relative vorticity field.
            - `'Q'` : A field defining an extra term in the vorticity equation,
              :math:`Q`. Defaults to a zero-valued field.

    dealias_field_keys : Iterable
        Keys for fields defined on the 'dealias' grid.
    prescribed_field_keys : Iterable
        Keys for fields defined on the 'base' grid which are prescribed, and
        which are not updated within :meth:`.step`. Defaults to `{'Q'}`.
    prescribed_dealias_field_keys : Iterable
        Keys for fields defined on the 'dealias' grid which are prescribed, and
        which are not updated within :meth:`.step`. Defaults to `set()`.
    """

    _defaults = {"L_x": required,  # x \in [-L_x, L_x]
                 "L_y": required,  # y \in [-L_y, L_y]
                 "N_x": required,
                 "N_y": required,
                 "beta": required,
                 "r": required,
                 "nu": required,
                 "dt": required}

    def __init__(self, parameters, *, field_keys=None, dealias_field_keys=None,
                 prescribed_field_keys=None, prescribed_dealias_field_keys=None):
        self._parameters = parameters = Parameters(parameters, defaults=self._defaults)
        if field_keys is None:
            field_keys = set()
        else:
            field_keys = set(field_keys)
        if dealias_field_keys is None:
            dealias_field_keys = set()
        else:
            dealias_field_keys = set(dealias_field_keys)
        field_keys.update({"psi", "zeta", "Q"})
        if prescribed_field_keys is None:
            prescribed_field_keys = {"Q"}
        if prescribed_dealias_field_keys is None:
            prescribed_dealias_field_keys = set()

        self._grid = grid = Grid(
            parameters["L_x"], parameters["L_y"],
            parameters["N_x"], parameters["N_y"])
        self._dealias_grid = dealias_grid = Grid(
            parameters["L_x"], parameters["L_y"],
            2 * parameters["N_x"], 2 * parameters["N_y"])

        self._fields = Fields(grid, field_keys)
        self._dealias_fields = Fields(dealias_grid, dealias_field_keys)
        self._prescribed_field_keys = set(prescribed_field_keys)
        self._prescribed_dealias_field_keys = set(prescribed_dealias_field_keys)

        self.zero_prescribed()
        self.initialize()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        def flatten(model):
            return model.flatten()

        def unflatten(aux_data, children):
            return cls.unflatten(aux_data, children)

        jax.tree_util.register_pytree_node(cls, flatten, unflatten)

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
        """Time step size."""

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
        """Base grid.
        """

        return self._grid

    @property
    def dealias_grid(self) -> Grid:
        """Dealias grid.
        """

        return self._dealias_grid

    @cached_property
    def dealias(self) -> SpectralGridTransfer:
        """Dealising utility object.
        """

        return SpectralGridTransfer(self.grid, self.dealias_grid)

    @property
    def parameters(self) -> Parameters:
        """Model parameters.
        """

        return self._parameters

    @property
    def fields(self) -> Fields:
        """Fields defined on the base grid.
        """

        return self._fields

    @property
    def dealias_fields(self) -> Fields:
        """Fields defined on the dealias grid.
        """

        return self._dealias_fields

    @cached_property
    def poisson_solver(self) -> PoissonSolver:
        """Solver for the Poisson equation.
        """

        return PoissonSolver(self.grid)

    def zero_prescribed(self):
        """Zero prescribed fields.
        """

        self.fields.zero(*self._prescribed_field_keys)
        self.dealias_fields.zero(*self._prescribed_dealias_field_keys)

    @abstractmethod
    def initialize(self, zeta=None):
        """Initialize the model.

        Parameters
        ----------

        zeta : :class:`jax.Array`
            Initial relative vorticity field. `None` indicates a zero-valued
            field.
        """

        self.fields.clear(keep_keys=self._prescribed_field_keys)
        self.dealias_fields.clear(keep_keys=self._prescribed_dealias_field_keys)
        self._n = 0

    @abstractmethod
    def step(self):
        """Take a timestep.
        """

        self._n += 1

    def ke(self):
        """The current kinetic energy, divided by density.

        Returns
        -------

        Real
            The current kinetic energy.
        """

        u = -self.fields["psi"] @ self.grid.D_y.T
        v = self.grid.D_x @ self.fields["psi"]
        return 0.5 * jnp.tensordot((u * u + v * v), self.grid.W)

    def ke_spectrum(self, N_x, N_y):
        """The current 2D kinetic energy spectrum.

        Computed using a type-I DST of the stream function, after interpolation
        onto a uniform grid. The spectrum is defined so that the sum of the
        spectrum equals, up to truncation, the current kinetic energy divided
        by the density.

        Parameters
        ----------

        N_x : Integral
            Number of intervals, in the :math:`x`-dimension, for the uniform
            grid.
        N_y : Integral
            Number of intervals, in the :math:`y`-dimension, for the uniform
            grid.

        Returns
        -------

        :class:`jax.Array`
            The current 2D kinetic energy spectrum.
        """

        x = jnp.linspace(-self.grid.L_x, self.grid.L_x, N_x + 1, dtype=self.grid.fdtype)
        y = jnp.linspace(-self.grid.L_y, self.grid.L_y, N_y + 1, dtype=self.grid.fdtype)
        psi = self.grid.interpolate(self.fields["psi"], x, y)
        I = jnp.outer(jnp.arange(N_x + 1, dtype=self.grid.idtype),
                      jnp.ones(N_y + 1, dtype=self.grid.idtype))
        J = jnp.outer(jnp.ones(N_x + 1, dtype=self.grid.idtype),
                      jnp.arange(N_y + 1, dtype=self.grid.idtype))
        k = jnp.pi * I / (2 * self.grid.L_x)
        l = jnp.pi * J / (2 * self.grid.L_y)
        return (0.5 * self.grid.L_x * self.grid.L_y
                * (k ** 2 + l ** 2) * (dst(dst(psi, axis=1), axis=0) ** 2))

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
                for key, value in lam_model.dealias_fields.items():
                    lam_model.dealias_fields[key] = value + zeta_model.dealias_fields[key]
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
        """Write solver to a :class:`zarr.hierarchy.Group`.

        Parameters
        ----------

        h : :class:`zarr.hierarchy.Group`
            Parent group.
        path : str
            Group path.

        Returns
        -------

        :class:`zarr.hierarchy.Group`
            Group storing the solver.
        """

        g = h.create_group(path)
        del h

        g.attrs["type"] = type(self).__name__
        g.attrs["n"] = int(self.n)
        self.parameters.write(g, "parameters")
        self.fields.write(g, "fields")
        self.dealias_fields.write(g, "dealias_fields")

        return g

    @classmethod
    def read(cls, h, path="solver"):
        """Read solver from a :class:`zarr.hierarchy.Group`.

        Parameters
        ----------

        h : :class:`zarr.hierarchy.Group`
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
        model = cls(Parameters.read(g, "parameters"))
        model.fields.update(Fields.read(g, "fields", grid=model.grid))
        model.dealias_fields.update(Fields.read(g, "dealias_fields", grid=model.dealias_grid))
        model.n = g.attrs["n"]

        return model

    def new(self):
        """Return a new :class:`.Solver` with the same configuration as this
        :class:`.Solver`.

        Returns
        -------

        :class:`.Solver`
            The new :class:`.Solver`.
        """

        model = type(self)(self.parameters)
        model.poisson_solver = self.poisson_solver
        return model

    def update(self, model):
        """Update the state of this :class:`.Solver`.

        Parameters
        ----------

        model : :class:`.Solver`
            Defines the new state of this :class:`.Solver`.
        """

        self.fields.update(model.fields)
        self.dealias_fields.update(model.dealias_fields)
        self.n = model.n

    def flatten(self):
        """Return a JAX flattened representation.

        Returns
        -------

        Sequence[Sequence[object, ...], Sequence[object, ...]]
        """

        return ((dict(self.fields), dict(self.dealias_fields), self.n),
                (self.parameters, self.poisson_solver))

    @classmethod
    def unflatten(cls, aux_data, children):
        """Unpack a JAX flattened representation.
        """

        parameters, poisson_solver = aux_data
        fields, dealias_fields, n = children

        model = cls(parameters)
        model.poisson_solver = poisson_solver
        model.fields.update({key: value for key, value in fields.items() if type(value) is not object})
        model.dealias_fields.update({key: value for key, value in dealias_fields.items() if type(value) is not object})
        if type(n) is not object:
            model.n = n

        return model

    def get_config(self):
        return {"type": type(self).__name__,
                "parameters": dict(self.parameters),
                "fields": dict(self.fields),
                "dealias_fields": dict(self.dealias_fields),
                "n": self.n}

    @classmethod
    def from_config(cls, config):
        config = {key: keras.saving.deserialize_keras_object(value) for key, value in config.items()}
        cls = cls._registry[config["type"]]
        model = cls(config["parameters"])
        model.fields.update(config["fields"])
        model.dealias_fields.update(config["dealias_fields"])
        model.n = config["n"]
        return model


class CNAB2Solver(Solver):
    """Chebyshev pseudospectral solver for the 2D barotropic vorticity
    equation on a beta-plane, using a CNAB2 time discretization.

    CNAB2 reference:

        - Uri M. Ascher, Steven J. Ruuth, and Brian T. R. Wetton,
        'Implicit-explicit methods for time-dependent partial differential
        equations', SIAM Journal on Numerical Analysis 32(3), 797--823, 1995,
        https://doi.org/10.1137/0732037

    Parameters
    ----------

    parameters : :class:`.Parameters`
        Model parameters. See :class:`.Solver`.
    """

    def __init__(self, parameters):
        super().__init__(
            parameters,
            field_keys={"F_1"},
            dealias_field_keys={"zeta", "u", "v"})

    @cached_property
    def modified_helmholtz_solver(self) -> ModifiedHelmholtzSolver:
        """Modified Helmholtz solver used for the implicit time discretization.
        """

        return ModifiedHelmholtzSolver(
            self.grid, alpha=(1 + 0.5 * self.dt * self.r), beta=0.5 * self.dt * self.nu)

    def initialize(self, zeta=None):
        super().initialize(zeta=zeta)
        if zeta is None:
            self.fields.zero("psi", "zeta")
            self.dealias_fields.zero("zeta", "u", "v")
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

        u_dg = self.dealias.to_higher_degree(-psi @ self.grid.D_y.T)
        v_dg = self.dealias.to_higher_degree(self.grid.D_x @ psi)
        zeta_dg = self.dealias.to_higher_degree(zeta)

        self.fields["psi"] = psi
        self.fields["zeta"] = zeta
        self.dealias_fields["zeta"] = zeta_dg
        self.dealias_fields["u"] = u_dg
        self.dealias_fields["v"] = v_dg

    def step(self):
        psi = self.fields["psi"]
        zeta = self.fields["zeta"]
        zeta_dg = self.dealias_fields["zeta"]
        u_dg = self.dealias_fields["u"]
        v_dg = self.dealias_fields["v"]

        F_0 = (
            - self.grid.D_x @ self.dealias.from_higher_degree(u_dg * zeta_dg)
            - self.dealias.from_higher_degree(v_dg * zeta_dg) @ self.grid.D_y.T
            - self.grid.D_x @ psi * self.beta
            + self.fields["Q"])
        G_0 = (
            self.nu * (self.grid.D_xx @ zeta + zeta @ self.grid.D_yy.T)
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

    def ke(self):
        u = self.dealias_fields["u"]
        v = self.dealias_fields["v"]
        return 0.5 * jnp.tensordot((u * u + v * v), self.dealias_grid.W)

    def steady_state_solve(self, *args, update=lambda model, *args: None, tol, max_it=10000):
        return super().steady_state_solve(*args, update=update, tol=tol, max_it=max_it, _min_n=1)

    def new(self):
        model = super().new()
        model.modified_helmholtz_solver = self.modified_helmholtz_solver
        return model

    def flatten(self):
        children, aux_data = super().flatten()
        return children, aux_data + (self.modified_helmholtz_solver,)

    @classmethod
    def unflatten(cls, aux_data, children):
        model = super().unflatten(aux_data[:-1], children)
        model.modified_helmholtz_solver = aux_data[-1]
        return model
