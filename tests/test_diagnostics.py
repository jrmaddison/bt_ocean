from bt_ocean.diagnostics import (
    FieldAverage, FieldProductAverage, Average, zero_point)
from bt_ocean.grid import Grid
from bt_ocean.model import Fields

import jax.numpy as jnp
from numpy import cbrt, sqrt
import pytest

from .test_base import eps


def test_field_average():
    L_x, L_y = 2.0, 3.0
    N_x, N_y = 5, 10
    grid = Grid(L_x, L_y, N_x, N_y)

    fields = Fields(grid, {"a", "b"})
    fields.zero(*fields.keys())
    avg = Average(grid, (FieldAverage("a"),))
    assert avg.w == 0

    fields["a"] = jnp.ones_like(fields["a"])
    avg.add(fields, weight=0.2)
    fields["a"] = jnp.full_like(fields["a"], 2)
    avg.add(fields, weight=0.3)

    avg_fields = avg.averaged_fields()
    assert abs(avg_fields["a"] - 1.6).max() == 0
    del avg_fields

    avg.zero()
    assert avg.w == 0


def test_field_product_average():
    L_x, L_y = 2.0, 3.0
    N_x, N_y = 5, 10
    grid = Grid(L_x, L_y, N_x, N_y)

    fields = Fields(grid, {"a", "b"})
    fields.zero(*fields.keys())
    avg = Average(grid, (FieldProductAverage("a", "b"),))
    assert avg.w == 0

    fields["a"] = jnp.ones_like(fields["a"])
    fields["b"] = jnp.ones_like(fields["b"])
    avg.add(fields, weight=0.2)
    fields["a"] = jnp.full_like(fields["a"], 2)
    fields["b"] = jnp.full_like(fields["b"], 5)
    avg.add(fields, weight=0.3)

    avg_fields = avg.averaged_fields()
    assert abs(avg_fields["a_b"] - 6.4).max() == 0
    del avg_fields

    avg.zero()
    assert avg.w == 0


@pytest.mark.parametrize("x_0", (-cbrt(2), 0, sqrt(2)))
@pytest.mark.parametrize("dx", (1, sqrt(3), -1, -sqrt(3)))
def test_zero_point(x_0, dx):
    x = jnp.array([x_0, x_0 + dx])

    y = jnp.array([1.0, -1.0])
    xz = zero_point(x, y, 0)
    assert abs(xz - x_0 - 0.5 * dx) < 10 * eps()

    y = jnp.array([-1.0, 1.0])
    xz = zero_point(x, y, 0)
    assert abs(xz - x_0 - 0.5 * dx) < 10 * eps()

    y = jnp.array([0.75, -0.25])
    xz = zero_point(x, y, 0)
    assert abs(xz - x_0 - 0.75 * dx) < 10 * eps()

    y = jnp.array([-0.75, 0.25])
    xz = zero_point(x, y, 0)
    assert abs(xz - x_0 - 0.75 * dx) < 10 * eps()
