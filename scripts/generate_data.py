import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from bt_ocean.diagnostics import (
    FieldAverage, FieldProductAverage, Average, zarr_append,
    DiagnosticsCsv, Timestep, Time, KineticEnergy, SeparationPoint,
    JetDiagnostics)
from bt_ocean.grid import Grid, SpectralGridTransfer
from bt_ocean.model import read_solver
from bt_ocean.precision import x64_enabled
from bt_ocean.timing import Timer

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numcodecs
import zarr

jax.config.update("jax_enable_x64", True)
zarr.storage.default_compressor = numcodecs.Blosc(cname="zstd", clevel=9, shuffle=1)
show_plots = False
if not show_plots:
    matplotlib.use("agg")

input_dir = pathlib.Path("input")
output_dir = pathlib.Path("output")

n0 = 3153600
with zarr.open(input_dir / f"{n0}.zarr", "r") as h:
    model = read_solver(h)
n_hour = int(3600 / model.dt + 0.5)
assert model.dt == 3600 / n_hour
n_day = 24 * n_hour
n_year = 365 * n_day
n_avg = 60 * n_day
N = n_year * 24

N_x_c = 64
N_y_c = 64
transfer = SpectralGridTransfer(
    Grid(model.grid.L_x, model.grid.L_y, N_x_c, N_y_c),
    model.grid)
h_c = zarr.open(output_dir / f"{N_x_c}x{N_y_c}.zarr", "w")
h_c.create_dataset("n", shape=(1, 0), dtype=transfer.grid_a.idtype)
h_c.create_dataset("t", shape=(1, 0), dtype=transfer.grid_a.fdtype)
for key in ("psi", "zeta"):
    h_c.create_dataset(key, shape=(transfer.grid_a.N_x + 1, transfer.grid_a.N_y + 1, 0),
                       dtype=transfer.grid_a.fdtype, chunks=(-1, -1, 1))
zarr_append(h_c["n"], (model.n,))
zarr_append(h_c["t"], (model.n * model.dt,))
zarr_append(h_c["psi"], transfer.from_higher_degree(model.fields["psi"]))
zarr_append(h_c["zeta"], transfer.from_higher_degree(model.fields["zeta"]))

avg = Average(
    model.dealias_grid,
    (FieldAverage("zeta"),
     FieldAverage("u"),
     FieldAverage("v"),
     FieldProductAverage("zeta", "zeta"),
     FieldProductAverage("u", "zeta"),
     FieldProductAverage("v", "zeta"),
     FieldProductAverage("u", "u"),
     FieldProductAverage("u", "v"),
     FieldProductAverage("v", "v")))
h_avg = zarr.open(output_dir / "avg.zarr", "w")

N_x_s = 1024
N_y_s = 1024
with x64_enabled():
    ke_spectrum = jnp.zeros((N_x_s + 1, N_y_s + 1), dtype=jnp.float64)
    ke_spectrum_w = 0.0
    h_spectrum = zarr.open(output_dir / "ke_spectrum_avg.zarr", "w")
    h_spectrum.create_dataset("w", shape=(1, 0), dtype=jnp.float64)
    h_spectrum.create_dataset("ke_spectrum", shape=(N_x_s + 1, N_y_s + 1, 0),
                              dtype=jnp.float64, chunks=(-1, -1, 1))

diag = DiagnosticsCsv(
    open(output_dir / "diagnostics.csv", "w"),
    (Timestep(), Time(), KineticEnergy(), SeparationPoint(),
     JetDiagnostics(x=-model.grid.L_x + jnp.cbrt(model.nu / model.beta), suffix="_munk"),
     JetDiagnostics(x=-model.grid.L_x + 100.0e3, suffix="_100km")))

assert model.n % n_hour == 0
assert model.n % n_day == 0
assert model.n % n_avg == 0
diag.write(model)
timer = Timer()
print(f"{model.n / n_day=} {timer.restart()=} {float(model.ke())=}", flush=True)
avg.zero()
while model.n < N:
    if model.n % n_avg == 0:
        avg.add(model.dealias_fields, weight=0.5)
        with x64_enabled():
            ke_spectrum = ke_spectrum + 0.5 * model.ke_spectrum(N_x_s, N_y_s)
            ke_spectrum_w += 0.5
    model.step()
    if model.n % n_hour == 0:
        diag.write(model)
    if model.n % n_avg == 0:
        avg.add(model.dealias_fields, weight=0.5)
        with x64_enabled():
            ke_spectrum = ke_spectrum + 0.5 * model.ke_spectrum(N_x_s, N_y_s)
            ke_spectrum_w += 0.5
    else:
        avg.add(model.dealias_fields, weight=1)
        with x64_enabled():
            ke_spectrum = ke_spectrum + model.ke_spectrum(N_x_s, N_y_s)
            ke_spectrum_w += 1
    if model.n % n_hour == 0:
        zarr_append(h_c["n"], (model.n,))
        zarr_append(h_c["t"], (model.n * model.dt,))
        zarr_append(h_c["psi"], transfer.from_higher_degree(model.fields["psi"]))
        zarr_append(h_c["zeta"], transfer.from_higher_degree(model.fields["zeta"]))
    if model.n % n_day == 0:
        print(f"{model.n / n_day=} {timer.restart()=} {float(model.ke())=}", flush=True)
    if model.n % (n_day * 5) == 0:
        plt.clf()
        plt.contourf(model.grid.x, model.grid.y, (model.fields["zeta"] + model.beta * model.grid.Y).T, 32)
        plt.gca().set_aspect(1)
        if show_plots:
            plt.show(block=False)
            plt.draw()
            plt.pause(0.1)
        plt.savefig(output_dir / f"{model.n}.png", dpi=576)
    if model.n % n_avg == 0:
        with zarr.open(output_dir / f"{model.n}.zarr", "w") as h:
            model.write(h)
        avg.append_averaged_fields(h_avg)
        avg.zero()
        with x64_enabled():
            zarr_append(h_spectrum["w"], (ke_spectrum_w,))
            zarr_append(h_spectrum["ke_spectrum"], ke_spectrum / ke_spectrum_w)
            ke_spectrum = jnp.zeros((N_x_s + 1, N_y_s + 1), dtype=jnp.float64)
            ke_spectrum_w = 0.0
if show_plots:
    plt.show()
