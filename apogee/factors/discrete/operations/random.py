import numpy as np


def random_factor(
    max_card=3, max_scope=3, max_factor=10, min_factor=1, ftype=lambda x: x
):
    n = np.random.randint(1, max_scope + 1)
    scope = apogee.legacy.random.urandint(min_factor, max_factor, size=(n,))
    card = np.random.randint(2, max_card + 1, n)
    return ftype(scope, card, np.ones(np.product(card)))


def random_factor_graph(n, gtype=lambda x: x, **kwargs):
    factors = []
    for i in range(n):
        factors.append(random_factor(**kwargs))
    return gtype(*factors)
