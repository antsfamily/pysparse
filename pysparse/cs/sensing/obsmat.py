#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-07-06 10:38:13
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
#
# @Note    : https://www.cnblogs.com/AndyJee/p/5091932.html
#
import numpy as np
from scipy.linalg import toeplitz, circulant, hankel


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    else:
        return v / norm


def column_normalize(A):
    n = A.shape[1]
    cols = np.hsplit(A, n)
    return np.hstack([normalize(col) for col in cols])


def gaussian0(shape, verbose=True):
    r"""generates Gauss observation matrix

    Generates M-by-N Gauss observation matrix

    .. math::
        {\bm \Phi} \sim {\mathcal N}(0, \frac{1}{M})

    Arguments
    --------------
    shape : `list` or `tuple`
        shape of Gauss observation matrix [M, N]

    Keyword Arguments
    ------------------
    verbose : `bool`
        display log info (default: {True})

    Returns
    -------------
    Phi : `ndarray`
        Gauss observation matrix :math:`\bm \Phi`.
    """

    (M, N) = shape
    if verbose:
        print("================in gauss================")
        print("===Construct Gauss observation matrix...")
        print("---Gauss observation matrix shape: ", shape)

    Phi = np.random.randn(M, N)  # (0, 1)
    Phi = np.sqrt(1.0 / M) * Phi  # (0, 1/M)

    if verbose:
        print("===Done!")

    return Phi


def bernoulli0(shape, verbose=True):
    r"""generates Bernoulli observation matrix

    Generates M-by-N Bernoulli observation matrix

    .. math::
        {\bm \Phi}_{ij} =\left\{\begin{array}{cc}{+\frac{1}{\sqrt{M}}} & {P=\frac{1}{2}} \\ 
                       {-\frac{1}{\sqrt{M}}} & {P=\frac{1}{2}}\end{array}=
                       \frac{1}{\sqrt{M}}\left\{\begin{array}{cc}{+1} & {P=\frac{1}{2}} \\ {-1} & {P=\frac{1}{2}}\end{array}\right.\right.

    Arguments
    ----------------
    shape : `list` or `tuple` 
        shape of Bernoulli observation matrix [M, N]

    Keyword Arguments
    -------------------

    verbose : `bool` 
        display log info (default: {True})

    Returns
    -----------------
    Phi : `ndarray`
        Bernoulli observation matrix
    """

    (M, N) = shape
    if verbose:
        print("================in bernoulli================")
        print("===Construct Bernoulli observation matrix...")
        print("---Bernoulli observation matrix shape: ", shape)

    Phi = np.random.randint(low=0, high=2, size=(M, N))  # (0, 1)
    Phi = np.sqrt(1.0 / M) * Phi  # (0, 1/M)

    # Phi = np.random.randint(low=-1, high=2, size=(M, N))  # (-1, 0, 1)
    # Phi = np.sqrt(3.0 / M) * Phi  # (0, 3/M)

    if verbose:
        print("===Done!")

    return Phi


def bernoulli(shape, verbose=True):
    r"""
    return a matrix, 
    which have bernoulli distribution elements
    columns are l2 normalized
    """

    return np.random.choice((0, 1), shape)


def gaussian(shape, verbose=True):
    r"""generates Gauss observation matrix

    Generates M-by-N Gauss observation matrix which have gaussian distribution elements(
    columns are l2 normalized).

    .. math::
        {\bm \Phi} \sim {\mathcal N}(0, \frac{1}{M})

    Arguments
    --------------
    shape : `list` or `tuple`
        shape of Gauss observation matrix [M, N]

    Keyword Arguments
    ------------------
    verbose : `bool`
        display log info (default: {True})

    Returns
    -------------
    A : `ndarray`
        Gauss observation matrix :math:`\bm A`.
    """

    m, n = shape
    A = np.random.randn(m, n)
    A = column_normalize(A)
    return A


def toeplitz(shape, verbose=True):
    r"""generates Toeplitz observation matrix

    Generates M-by-N Toeplitz observation matrix

    .. math::
        {\bm \Phi}_{ij} = \left[\begin{array}{ccccc}{a_{0}} & {a_{-1}} & {a_{-2}} & {\cdots} & {a_{-n+1}} \\ {a_{1}} & {a_{0}} & {a_{-1}} & {\cdots} & {a_{-n+2}} \\ {a_{2}} & {a_{1}} & {a_{0}} & {\cdots} & {a_{-n+3}} \\ {\vdots} & {\vdots} & {\vdots} & {\ddots} & {\vdots} \\ {a_{n-1}} & {a_{n-2}} & {a_{n-3}} & {\cdots} & {a_{0}}\end{array}\right]

    Arguments
    ------------
    shape : `list` or `tuple` 
        shape of Toeplitz observation matrix [M, N]

    Keyword Arguments
    ----------------------
    verbose : `bool`
        display log info (default: {True})

    Returns
    -------------
    A : `ndarray` 
        Toeplitz observation matrix :math:`\bm A`.
    """

    (M, N) = shape
    if verbose:
        print("================in toeplitz================")
        print("===Construct Toeplitz observation matrix...")
        print("---Toeplitz observation matrix shape: ", shape)

    if verbose:
        print("===Done!")

    return Phi


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    Phi = gaussian((32, 256), verbose=True)
    # Phi = bernoulli((32, 256), verbose=True)
    print(Phi)

    plt.figure()
    plt.imshow(Phi)
    plt.show()
