"""
The MIT License

Copyright (c) 2017-2020 Mark Douthwaite

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np


def scale(*args, **kwargs):
    """Scale an array. """

    # alternative signature.
    return normalise(*args, **kwargs)


def normalise(
    x: np.ndarray,
    a_min: np.float32 = -np.inf,
    a_max: np.float32 = np.inf,
    method: str = "default",
    **kwargs
):
    """
    Normalise an array 'x'.

    Parameters
    ----------
    x: array_like
        The input array to be normalised.
    lower: float
        The lower bound of the array -- array will be clipped at this value.
    upper: float
        The upper bound of the array -- array will be clipped at this value.
    method: str
        The method of normalisation to be applied.
        'default' - Apply a normalisation transformation such that the array sums to 1.
        'standard' - Apply standardisation transformation to the array by scaling
                     according to mean
        and std.
        'mean' - Apply a mean normalisation transformation to the array.
        'scale' - Scale the array between specified arbitary bounds 'a' and 'b'.
        'spectral' - Scale the array such that it has unit spectral radius, then apply
                     arbitrary
        scaling factor.

    kwargs: see methods

    Returns
    -------
    out: array_like
        The normalised input array.

    Examples
    --------
    >>> x = np.array([0.35, 0.97, 0.04, 0.76])
    >>> normalise(x)
    [ 0.16509434  0.45754717  0.01886792  0.35849057]
    >>> x = np.array([[0.9, 0.6], [0.1, 0.2]])
    >>> normalise(x)  # entire array sums to 1.
    [[ 0.47368421  0.31578947]
     [ 0.05263158  0.15789474]]
    >>> normalise(x, axis=1), # 'row-wise' normalisation (in this case)
    [[ 0.6   0.4 ]
     [ 0.25  0.75]]
    >>> normalise(x, axis=0)  # 'column-wise' normalisation (in this case)
    [[ 0.9         0.66666667]
     [ 0.1         0.33333333]]

    """

    x = np.asanyarray(x)
    x = np.clip(x, a_min, a_max)

    # ugh
    if method == "prob":  # deprecated
        return norm(x, **kwargs)
    elif method == "default":
        return norm(x, **kwargs)
    elif method == "scale":
        return abs_norm(x, **kwargs)
    elif method == "standard":
        return std_norm(x, **kwargs)
    elif method == "mean":
        return mean_norm(x, **kwargs)
    elif method == "spectral":
        return spectral_norm(x, **kwargs)
    else:
        raise ValueError("Unknown normalisation method '{0}'.".format(method))


def spectral_norm(x, a=1.0, axis=None):
    """
    Scale a (square) array such that it has unit spectral radius multiplied by an arbitrary real
    scale parameter.

    """

    r = a / np.max(np.abs(np.linalg.eigvals(x)), axis=axis)
    return r * x


def norm(x, axis=None):
    """
    Normalise a given array such that the sum of the resulting array is one.

    Parameters
    ----------
    x: array_like
        The array to be normalised.
    axis: None or int or tuple of ints, optional
        Axis or axes along which the normalisation is computed.

    Returns
    -------
    out: ndarray
        An array of the same dimension passed in, normalised to sum to one.

    """
    return np.true_divide(x, np.sum(x, axis=axis, keepdims=True))


def abs_norm(x, a=0, b=1, axis=None):
    """
    Normalise an array to values between specified range (defaults to between 0 and 1).

    (Feature scaling, unity-normalisation)

    Parameters
    ----------
    x: array_like
        The array to be normalised.
    a: number
        The lower normalisation bound. (i.e. minimum value of the resulting array)
    b: number
        The upper normalisation bound. (i.e. maximum value of the resulting array)
    axis: None or int or tuple of ints, optional
        Axis or axes along which the normalisation is computed.

    Returns
    -------
    out: array_like
        An array of the same dimension passed in, normalised over given range.

    Examples
    --------
    >>> x = np.array([0.35, 0.97, 0.04, 0.76])
    >>> abs_normalise(x)
    [ 0.33333333  1.          0.          0.77419355]

    """

    _max = np.max(x, axis=axis, keepdims=True)
    _min = np.min(x, axis=axis, keepdims=True)
    return a + ((x - _min) * (b - a)) / (_max - _min)


def std_norm(x, axis=None):
    """
    Normalise a distribution according to standard _statistics (mean & variance).

    (Standard score, standardisation)

    Parameters
    ----------
    x: ndarray
        The array to be normalised.

    Returns
    -------
    out: ndarray
        An array of the same dimension passed in, array will have zero-mean.

    Examples
    --------
    >>> x = np.array([0.35, 0.97, 0.04, 0.76])
    >>> std_norm(x)
    [ 0.33333333  1.          0.          0.77419355]

    """

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    scaled = np.true_divide(np.subtract(x, mean), std)
    return scaled


def mean_norm(x, axis=None):
    """Apply mean normalisation to input array."""

    _mean = np.mean(x, axis=axis, keepdims=True)
    _max = np.max(x, axis=axis)
    _min = np.min(x, axis=axis)
    return (x - _mean) / (_max - _min)
