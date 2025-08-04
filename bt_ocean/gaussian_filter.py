"""Gaussian filtering.
"""

from collections.abc import Sequence
from functools import partial

import jax
import jax.numpy as jnp

__all__ = \
    [
        "pad",

        "fftconvolve_1d",

        "gaussian_filter_kernel_1d",
        "gaussian_filter_1d",
        "gaussian_filter"
    ]


@partial(jax.jit, static_argnames={"pad_width", "mode"})
def pad(input, pad_width, *, mode="constant", cval=0):
    """Array padding.

    Parameters
    ----------

    input : :class:`jax.Array`
        Array to pad.
    pad_width
        As for :func:`jax.numpy.pad`.
    mode : str
        Padding type. As for :func:`scipy.ndimage.gaussian_filter1d` (SciPy
        1.16.1).
    cval : Complex
        Value to use with `'constant'` mode.

    :class:`jax.Array`
        Padded array.
    """

    mode = {"reflect": "symmetric",
            "grid-mirror": "symmetric",
            "constant": "constant",
            "grid-constant": "constant",
            "nearest": "edge",
            "mirror": "reflect",
            "wrap": "wrap",
            "grid-wrap": "wrap"}[mode]
    if mode == "constant":
        kwargs = {"constant_values": cval}
    elif mode in {"symmetric", "reflect"}:
        kwargs = {"reflect_type": "even"}
    else:
        kwargs = {}

    return jnp.pad(input, pad_width, mode=mode, **kwargs)


@partial(jax.jit, static_argnames={"mode", "axis"})
def fftconvolve_1d(input, kernel, *, mode="constant", cval=0, axis=-1):
    """1D FFT-based convolution.

    Performs a linear convolution via a discrete Fourier transform approach.
    See section 1.4 in

        - 'Advanced Digital Signal Processing', John. G. Proakis, Charles M.
          Rader, Fuyun Ling, and Chrysostomos L. Nikias, Macmillan Publishing
          Company, 1992

    Implementation details:

        - Boundary conditions are applied by extending the input.
        - Additional zero padding is then used so that FFTs are performed on
          inputs with a size of an exact power of two. This avoids accuracy
          loss on some systems.

    Parameters
    ----------

    input : :class:`jax.Array`
        Data to filter. Must have real dtype.
    kernel : :class:`jax.Array`
        1D convolution kernel. Must have odd length.
    mode : str
        Boundary conditions. As for :func:`scipy.ndimage.gaussian_filter1d`
        (SciPy 1.16.1).
    cval : Real
        Value to use with `'constant'` mode.
    axis : Integral
        Axis along which to apply the filter.

    Returns
    -------

    :class:`jax.Array`
        Result of the convolution.
    """

    input = jnp.swapaxes(input, axis, -1)

    N = input.shape[-1]
    K, = kernel.shape
    if K % 2 == 0:
        # Require caller to resolve ambiguity, catches K = 0
        raise ValueError("Odd kernel size required")
    # Kernel 'radius'
    K = (K - 1) // 2

    # Boundary conditions
    input_e = pad(input, ((0, 0),) * (len(input.shape) - 1) + ((K, K),),
                  mode=mode, cval=cval)

    # Zero pad up to an exact power of two
    n = 1
    while n < max(kernel.shape[0], input_e.shape[-1]):
        n *= 2
    input_e = jnp.zeros_like(input_e, shape=input_e.shape[:-1] + (n,)).at[..., :input_e.shape[-1]].set(input_e)
    kernel_e = jnp.zeros_like(kernel, shape=(n,)).at[:kernel.shape[0]].set(kernel)

    # Minimum size required, see section 1.4.2 in Proakis et al 1992 (full
    # reference in docstring)
    assert n >= N + kernel.shape[0] - 1
    input_s = jnp.fft.rfft(input_e, axis=-1)
    kernel_s = jnp.fft.rfft(kernel_e, axis=-1)
    # Using NumPy 'general broadcasting rules'
    #    https://numpy.org/doc/stable/user/basics.broadcasting.html
    #    [accessed 2025-06-02]
    output_s = input_s * kernel_s
    # Defensively provide n here, needed for n odd
    output_e = jnp.fft.irfft(output_s, n=n, axis=-1)
    assert output_e.shape == input_e.shape

    # Shift by K for the boundary conditions, and by K to center the kernel
    output = output_e[..., 2 * K:2 * K + N]
    assert output.shape == input.shape

    output = jnp.swapaxes(output, -1, axis)
    return output


@partial(jax.jit, static_argnames={"sigma", "truncate", "radius"})
def gaussian_filter_kernel_1d(sigma, *, truncate=4, radius=None):
    """Construct a 1D Gaussian filter kernel. Normalized to have unit sum after
    truncation.

    Parameters
    ----------

    sigma : Real
        Filter standard deviation.
    truncate : Real
        Defines the filter 'radius', `round(sigma * truncate)`. Ignored if
        `radius` is supplied.
    radius : Real
        Defines the kernel filter size, `2 * round(radius) + 1`. `truncate`
        is ignored if supplied.

    Returns
    -------

    :class:`jax.Array`
        The filter kernel.
    """

    if radius is None:
        radius = sigma * truncate
    radius = round(radius)
    if radius < 0:
        raise ValueError("Invalid radius")

    i0 = radius
    i = jnp.arange(2 * radius + 1, dtype=int)
    kernel = jnp.exp(-((i - i0) ** 2) / (2 * (sigma ** 2)))
    # Normalize so that the filter is constant preserving (up to boundary
    # conditions)
    kernel = kernel / kernel.sum()

    return kernel


@partial(jax.jit, static_argnames={"sigma", "mode", "truncate", "radius", "axis"})
def gaussian_filter_1d(input, sigma, *,
                       mode="reflect", cval=0, truncate=4, radius=None, axis=-1):
    """Apply a 1D Gaussian convolutional filter. API intended to be similar to
    :func:`scipy.ndimage.gaussian_filter1d` (SciPy 1.16.1).

    Parameters
    ----------

    input : :class:`jax.Array`
        Data to filter. Must have real dtype.
    sigma : Real
        Filter standard deviation.
    mode : str
        Boundary conditions. As for :func:`scipy.ndimage.gaussian_filter1d`
        (SciPy 1.16.1).
    cval : Real
        Value to use with `'constant'` mode.
    truncate : Real
        Defines the filter 'radius', `round(sigma * truncate)`. Ignored if
        `radius` is supplied.
    radius : Real
        Defines the kernel filter size, `2 * round(radius) + 1`. `truncate`
        is ignored if supplied.
    axis : Integral
        Axis along which to apply the filter.

    Returns
    -------

    :class:`jax.Array`
        Filtered data.
    """

    kernel = gaussian_filter_kernel_1d(sigma, truncate=truncate, radius=radius)
    return fftconvolve_1d(input, kernel, mode=mode, cval=cval, axis=axis)


@partial(jax.jit, static_argnames={"sigma", "mode", "truncate", "radius", "axes"})
def gaussian_filter(input, sigma, *,
                    mode="reflect", cval=0, truncate=4, radius=None, axes=None):
    """Apply a Gaussian convolutional filter. API intended to be similar to
    :func:`scipy.ndimage.gaussian_filter` (SciPy 1.16.1).

    Parameters
    ----------

    input : :class:`jax.Array`
        Data to filter. Must have real dtype.
    sigma : Real or Sequence[Real, ...]
        Filter standard deviations.
    mode : str
        Boundary conditions. As for :func:`scipy.ndimage.gaussian_filter`
        (SciPy 1.16.1).
    cval : Real
        Value to use with `'constant'` mode.
    truncate : Real
        Defines the filter 'radius', `round(sigma * truncate)`. Can be
        overridden by `radius`.
    radius : Real or Sequence[None or Real, ...]
        Defines the kernel filter size, `2 * round(radius) + 1`. Overrides
        `truncate` if not `None`.
    axes : Integral or Sequence[Integral, ...]
        Axes along which to apply the filter.

    Returns
    -------

    :class:`jax.Array`
        Filtered data.
    """

    def map_axis(u, axis):
        if axis < 0:
            axis += len(u.shape)
        if axis < 0 or axis >= len(u.shape):
            raise ValueError("axis out of bounds")
        return axis

    if axes is None:
        axes = tuple(range(len(input.shape)))
    elif not isinstance(axes, Sequence):
        axes = (axes,)
    axes = tuple(map(partial(map_axis, input), axes))
    if len(set(axes)) != len(axes):
        raise ValueError("Duplicated axis")
    if not isinstance(sigma, Sequence):
        sigma = tuple(sigma for _ in axes)
    if radius is None:
        radius = tuple(None for _ in axes)
    elif not isinstance(radius, Sequence):
        radius = tuple(radius for _ in axes)

    if len(sigma) != len(axes):
        raise ValueError("Invalid sigma")
    if len(radius) != len(axes):
        raise ValueError("Invalid radius")

    output = input.copy()
    for axis, sigma_i, radius_i in zip(axes, sigma, radius):
        output = gaussian_filter_1d(
            output, sigma_i, truncate=truncate, mode=mode, cval=cval, radius=radius_i, axis=axis)
    return output
