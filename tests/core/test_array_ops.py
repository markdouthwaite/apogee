import numpy as np
from apogee.core import array_index


def test_array_index_1d():
    a = [[1, 2], [3, 4]]
    b = [[0, 1], [1, 2], [3, 4]]

    assert np.all(array_index(a, b) == [1, 2])


def test_array_sort():
    pass

