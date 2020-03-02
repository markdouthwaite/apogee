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


def entropy(p: np.ndarray, **kwargs):
    """Compute the entropy of a vector."""

    p = np.clip(p, 1e-16, 1.0 - 1e-16)
    return -np.sum(p * (np.log(p)), **kwargs)


def relative_entropy(p: np.ndarray, q: np.ndarray, eps: float = 1e-16, **kwargs):
    """Compute the Relative Entropy of two vectors."""

    p = np.clip(p, eps, 1.0 - eps)
    q = np.clip(q, eps, 1.0 - eps)
    return np.sum(p * (np.log(p / q)), **kwargs)


def cross_entropy(p: np.ndarray, q: np.ndarray, **kwargs):
    """Compute the cross entropy of two discrete probability distributions."""

    return entropy(p, **kwargs) - relative_entropy(p, q, **kwargs)


def symmetric_relative_entropy(p: np.ndarray, q: np.ndarray, *args, **kwargs):
    """Compute the symmetric of relative entropy (K-L Divergence)."""

    return relative_entropy(p, q, *args, **kwargs) + relative_entropy(
        q, p, *args, **kwargs
    )


def kullback_leibler_divergence(*args, **kwargs):
    """Compute the Kullback-Liebler Divergence of two vectors."""

    return relative_entropy(*args, **kwargs)


def symmetric_kullback_leibler_divergence(*args, **kwargs):
    """Compute the symmetric of relative entropy (K-L Divergence). (Wrapper)"""

    return symmetric_relative_entropy(*args, **kwargs)


def discrete_mutual_information(joint: np.ndarray):
    """
    Compute the mutual information of a discrete joint probability distribution (JPD).

    Notes
    -----
    * Derived from sklearn.metrics.cluster.supervised.mutual_info_score

    """

    joint *= 100  # hack, ugh.
    nzx, nzy = np.nonzero(joint)
    nz_val = joint[nzx, nzy]
    nzx, nzy = np.nonzero(joint)
    contingency_sum = joint.sum()
    pi = np.ravel(joint.sum(axis=1))
    pj = np.ravel(joint.sum(axis=0))
    log_contingency_nm = np.log(nz_val)
    contingency_nm = nz_val / contingency_sum
    outer = pi.take(nzx) * pj.take(nzy)
    log_outer = -np.log(outer) + np.log(pi.sum()) + np.log(pj.sum())
    mi = (
        contingency_nm * (log_contingency_nm - np.log(contingency_sum))
        + contingency_nm * log_outer
    )
    return mi.sum()


def normalised_discrete_mutual_information(joint: np.ndarray, **kwargs):
    """Compute the discrete, normalised mutual information score for a distribution."""

    z = discrete_mutual_information(joint, **kwargs)
    return z / np.log(len(joint))


def mutual_information_index(xi, y, normed=True):
    """
    Compute the Mutual Information Index for a distribution.

    References
    ----------
    1)  Critchfield, G. C., Willard, K. E. & Connelly, D. P. 1986
        Probabilistic sensitivity analysis methods for general
        decision models. Comp. Biomed. Res. 19, 254–265

    """

    return (
        (entropy(y) - entropy(xi)) / entropy(y) if normed else entropy(y) - entropy(xi)
    )


def gaussian_mutual_information(a: np.ndarray, b: np.ndarray):
    """Compute the Gaussian Mutual Information given two vectors."""

    saa = np.atleast_2d(np.cov(a))
    sbb = np.atleast_2d(np.cov(b))
    s = np.atleast_2d(np.cov(np.array(np.c_[a, b]).T))
    return 0.5 * np.log((np.linalg.det(saa) * np.linalg.det(sbb)) / np.linalg.det(s))
