import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from bt_ocean.diagnostics import DiagnosticsCsv, Timestep, Time, KineticEnergy
from bt_ocean.model import CNAB2Solver
from bt_ocean.parameters import n_hour, n_day, n_year, parameters, Q
from bt_ocean.timing import Timer

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numcodecs
import numpy as np
import zarr

jax.config.update("jax_enable_x64", True)
zarr.storage.default_compressor = numcodecs.Blosc(cname="zstd", clevel=9, shuffle=1)
show_plots = False
if not show_plots:
    matplotlib.use("agg")
np.random.seed()
seed = np.random.randint(2 ** 32)
print(f"{seed=}", flush=True)
np.random.seed(seed)

output_dir = pathlib.Path("output")

model = CNAB2Solver(parameters)
N = n_year * 12

model.fields["Q"] = Q(model.grid)
model.initialize(jnp.array(model.beta * model.grid.L_y
                           * np.random.standard_normal((model.grid.N_x + 1, model.grid.N_y + 1))
                           / jnp.sqrt(model.grid.W), dtype=model.grid.fdtype))
with zarr.open(output_dir / f"{model.n}.zarr", "w") as h:
    model.write(h)

diag = DiagnosticsCsv(
    open(output_dir / "diagnostics.csv", "w"),
    (Timestep(), Time(), KineticEnergy()))

assert model.n % n_hour == 0
assert model.n % n_day == 0
assert model.n % n_year == 0
diag.write(model)
timer = Timer()
print(f"{model.n / n_day=} {timer.restart()=} {float(model.ke())=}", flush=True)
while model.n < N:
    model.step()
    if model.n % n_hour == 0:
        diag.write(model)
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
    if model.n % n_year == 0:
        with zarr.open(output_dir / f"{model.n}.zarr", "w") as h:
            model.write(h)
if show_plots:
    plt.show()
