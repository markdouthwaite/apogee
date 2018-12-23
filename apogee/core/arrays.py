import itertools
import numpy as np
from functools import reduce


def sort(arr, reverse=True, **kwargs):
    x = np.sort(arr, **kwargs)
    if reverse:
        x = x[::-1]
    return x


def union(*args):
    return reduce(np.union1d, args)


def difference(*args):
    return reduce(np.setdiff1d, args)


def intersect(*args):
    return reduce(np.intersect1d, args)


def equals(a, b):
    if np.all(np.array(a) == np.array(b)):
        return True
    else:
        return False


def contains(*args):
    if len(intersect(*args)) > 0:
        return True
    else:
        return False


def subset(a, b):
    if len(a) == len(intersect(a, b)):
        return True
    else:
        return False


def array_map(*args, **kwargs):
    return np.asarray(list(map(*args, **kwargs)))


def array_reduce(*args, **kwargs):
    return np.asarray(list(reduce(*args, **kwargs)))


def array_index(a, b):
    a = np.asarray(a).tolist()
    b = np.asarray(b).tolist()

    output = []
    for x in a:
        output.append(b.index(x))

    return np.asarray(output, dtype=np.int32)


def array_mapping(a, b):
    return np.where(np.asarray(b)[:, None] == np.asarray(a)[None, :])[1]


def index_map(a: np.ndarray, b: np.ndarray):
    return np.where(a[:, None] == b[None, :])[1]


def index_map_1d(a: np.ndarray, b: np.ndarray, **kwargs):
    return np.where(np.in1d(a, b, **kwargs))[0]


def ndarange(*args, shape=None, **kwargs):
    arr = np.array([np.arange(*args[i], **kwargs) for i in range(len(args))])
    return arr.reshape(shape) if shape is not None else arr.T


def cartesian_product(*arr) -> np.ndarray:
    return np.asarray(list(itertools.product(*arr)))

