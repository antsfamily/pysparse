#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-06-13 10:34:43
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
import numpy as np
from pysparse.utils.const import *


def dftmat(N):
    r"""Discrete Fourier transform matrix

    .. math::
       {\bm y} = {\bm D}{\bm x}
       :label: equ-DFT_MatrixRep

    where, :math:`{\bm x} = (x_n)_{N\times 1}, x_n = x[n]`, :math:`{\bm D} = (d_{ij})_{N\times N}` can be expressed as

    .. math::
       {\bm D} = \sqrt{\frac{2}{N}}\left[\begin{array}{cccc}{1/\sqrt{2}} & {1/\sqrt{2}} & {1/\sqrt{2}} & {\cdots} & {1/\sqrt{2}} \\ {\cos \frac{\pi}{2 N}} & {\cos \frac{3 \pi}{2 N}} & {\cos \frac{5 \pi}{2 N}} & {\cdots} & {\cos \frac{(2 N-1) \pi}{2 N}} \\ {\vdots} & {\vdots} & {\vdots} & {\vdots} \\ {\cos \frac{(N-1) \pi}{2 N}} & {\cos \frac{3(N-1) \pi}{2 N}} & {\cos \frac{5(N-1) \pi}{2 N}} & {\cdots} & {\cos \frac{(2 N-1)(N-1) \pi}{2 N}}\end{array}\right]
       :label: equ-DFT_Matrix

    Arguments
    ----------------
    N : integer
        signal dimesion.

    Returns
    -------------------
    T : ndarray
        DFT matrix.
    """

    # r, c = np.mgrid[0:N, 0:N]

    # T = np.sqrt(2 / N) * np.cos(PI * (2 * c + 1) * r / (2 * N))
    # T[0, :] = T[0, :] / np.sqrt(2)

    T = np.fft.fft(np.eye(N)) / np.sqrt(N * 1.0)

    return T


def dft1(x, axis=0):
    r"""1-Dimension Discrete cosine transform

       The DFT of signal :math:`x[n], n=0, 1,\cdots, N-1` is expressed as

       .. math::
          y[k] = {\rm DFT}(x[n]) = \left\{ {\begin{array}{lll}
              {\sqrt{\frac{2}{N}}\sum_{n=0}^{N-1}x[n]\frac{1}{\sqrt 2}, \quad k=0}\\
              {\sqrt{\frac{2}{N}}\sum_{n=0}^{N-1}x[n]{\rm cos}\frac{(2n + 1)k\pi}{2N}, \quad k=1, \cdots, N-1}
              \end{array}} \right.
          :label: equ-DFT

       where, :math:`k=0, 1, \cdots, N-1`

    N. Ahmed, T. Natarajan, and K. R. Rao. Discrete cosine transform.
    IEEE Transactions on Computers, C-23(1):90–93, 1974. doi:10.1109/T-C.1974.223784

    Arguments
    -------------
    x : numpy array
        signal vector or matrix

    Keyword Arguments
    --------------------
    axis : number
        transformation axis when x is a matrix (default: {0}, col)

    Returns
    -----------
    y : numpy array
        the coefficients.


    """

    x = np.array(x)
    if np.ndim(x) > 1:
        N = np.size(x, axis=axis)
        T = dftmat(N)
        if axis is 0:
            return np.matmul(T, x)
        if axis is 1:
            return np.matmul(T, x.transpose()).transpose()
    if np.ndim(x) is 1:
        N = np.size(x)
        T = dftmat(N)
        return np.matmul(T, x)


def idft1(y, axis=0):
    r"""1-Dimension Inverse Discrete cosine transform

    .. math::
       {\bm x} = {\bm D}^{-1}{\bm y} = {\bm D}^T{\bm y}
       :label: equ-IDFT_MatrixRep

    Arguments
    -------------
    y : numpy array
        coefficients

    Keyword Arguments
    ------------------
    axis : number
        IDFT along which axis (default: {0})

    Returns
    -------------
    x : numpy array
        recovered signal.
    """

    y = np.array(y)
    if np.ndim(y) > 1:
        N = np.size(y, axis=axis)
        # T = np.linalg.inv(dftmat(N))
        T = dftmat(N).transpose()
        if axis is 0:
            return np.matmul(T, y)
        if axis is 1:
            return np.matmul(T, y.transpose()).transpose()
    if np.ndim(y) is 1:
        N = np.size(y)
        T = dftmat(N)
        return np.matmul(T, y)


def dft2(X):
    r"""2-Dimension Discrete cosine transform

    dft1(dft1(X, axis=0), axis=1)

    Arguments
    -----------------
    X : numpy array
        signal matrix

    Returns
    -----------
    Y : numpy array
        coefficients matrix
    """

    return dft1(dft1(X, axis=0), axis=1)


def idft2(X):
    r"""2-Dimension Inverse Discrete cosine transform

    idft1(idft1(X, axis=0), axis=1)

    Arguments
    --------------
    X : numpy array
        signal matrix

    Returns
    --------------
    Y : numpy array
        coefficients matrix
    """

    return idft1(idft1(X, axis=0), axis=1)


def dftdict(N, isnorm=False, verbose=False):
    r"""Complete DFT dictionary


    .. math::
       {\bm D} = \sqrt{\frac{2}{N}}\left[\begin{array}{cccc}{1/\sqrt{2}} & {1/\sqrt{2}} & {1/\sqrt{2}} & {\cdots} & {1/\sqrt{2}} \\ {\cos \frac{\pi}{2 N}} & {\cos \frac{3 \pi}{2 N}} & {\cos \frac{5 \pi}{2 N}} & {\cdots} & {\cos \frac{(2 N-1) \pi}{2 N}} \\ {\vdots} & {\vdots} & {\vdots} & {\vdots} \\ {\cos \frac{(N-1) \pi}{2 N}} & {\cos \frac{3(N-1) \pi}{2 N}} & {\cos \frac{5(N-1) \pi}{2 N}} & {\cdots} & {\cos \frac{(2 N-1)(N-1) \pi}{2 N}}\end{array}\right]
       :label: equ-DFT_Matrix

    Because :math:`{\bm z} = {\bm D}{\bm x}`, :math:`{\bm D}{\bm D}^T = {\bm I}`
    So, :math:`{\bm x} = {\bm D}^T{\bm z}`

    Arguments
    -------------
    N : integer
        The dictionary is of size :math:`N\times N`

    Returns
    -------------
    D : numpy array
        DFT dictionary ( :math:`{\bm D}^T` ).
    """
    if verbose:
        print("================in dftdict================")

    D = dftmat(N)

    if verbose:
        print('---Done!')

    return D.transpose()


def odftdict(dictshape, isnorm=False, verbose=False):
    r"""Overcomplete 1D-DFT dictionary

    .. math::
       {\bm D} = \sqrt{\frac{2}{N}}\left[\begin{array}{cccc}{1/\sqrt{2}} & {1/\sqrt{2}} & {1/\sqrt{2}} & {\cdots} & {1/\sqrt{2}} \\ {\cos \frac{\pi}{2 N}} & {\cos \frac{3 \pi}{2 N}} & {\cos \frac{5 \pi}{2 N}} & {\cdots} & {\cos \frac{(2 N-1) \pi}{2 N}} \\ {\vdots} & {\vdots} & {\vdots} & {\vdots} \\ {\cos \frac{(M-1) \pi}{2 N}} & {\cos \frac{3(M-1) \pi}{2 N}} & {\cos \frac{5(M-1) \pi}{2 N}} & {\cdots} & {\cos \frac{(2 N-1)(M-1) \pi}{2 N}}\end{array}\right]
       :label: equ-ODFT_Matrix

    .. math::
       {\bm D} = \left[\frac{{\bm d}_0}{\|{\bm d}_0\|_2}, \frac{{\bm d}_1}{\|{\bm d}_1\|_2},\cdots, \frac{{\bm d}_{N-1}}{\|{\bm d}_{N-1}\|_2}\right]
       :label: equ-ODFT_Matrix_normed

    Arguments
    ----------------------
    dictshape : tuple
        dictionary shape

    Keyword Arguments
    ----------------------
    isnorm : bool
        normlize atoms (default: {False})
    verbose : bool
        display log (default: {True})
    """

    if verbose:
        print("================in odftdict================")

    M, N = dictshape
    N, M = dictshape

    if verbose:
        print("---Construct Overcomplete 1D-DFT dictionary...")

    r, c = np.mgrid[0:M, 0:N]

    D = np.sqrt(2 / N) * np.cos(PI * (2 * c + 1) * r / (2 * N))
    D[0, :] = D[0, :] / np.sqrt(2)

    if isnorm:
        if verbose:
            print("---Normalization...")
        for k in range(N):
            v = D[:, k]
            v = v - np.mean(v)
            norm = np.linalg.norm(v)
            D[:, k] = v / norm

    if verbose:
        print("---Done!")

    return D.transpose()


def odftndict(dictshape, axis=-1, isnorm=False, verbose=False):
    r"""generates Overcomplete nD-DFT dictionary

    .. math::
       {\bm D}_{nd} = {\bm D}_{(n-1)d} \otimes {\bm D}_{(n-1)d}.
       :label: equ-CreatenDDFT_Matrix

    Arguments
    ---------------------
    dictshape : `list` or `tuple`
        shape of DFT dict [M, N]

    Keyword Arguments
    ---------------------
    axis : `number`
        Axis along which the dft is computed. If -1 then the transform
        is multidimensional(default=-1) (default: {-1})

    isnorm : `bool`
        normlize atoms (default: {False})

    verbose : `bool`
        display log info (default: {True})

    Returns
    ---------------------
    OD : `ndarray`
        Overcomplete nD-DFT dictionary
    """

    if verbose:
        print("================in odftndict================")
    (M, N) = dictshape
    if verbose:
        print("===Construct Overcomplete nD-DFT dictionary...")
        print("---DFT dictionary shape: ", dictshape)

    MM, NN = dictshape
    M = int(np.sqrt(MM))
    N = int(np.sqrt(NN))

    D = odftdict((M, N), isnorm=isnorm)
    D = np.kron(D, D)

    if verbose:
        print("---Done!")
    return D


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import pysparse as pys
    import scipy.io as sio

    x = [0, 1, 2, 3, 4, 5]
    y = dft1(x)
    print(y)

    print("----------------------")
    print(dftmat(3))
    print(dftdict(3))
    T = dftmat(6)
    print(np.matmul(T, T.transpose()))

    x = [[0, 1, 2], [3, 4, 5]]

    print("----------------------")
    y = dft1(x, axis=0)  # Matlab--> dft(x)
    print(y)
    y = dft1(x, axis=1)  # Matlab--> dft(x')'
    print(y)

    print("----------------------")
    y = dft2(x)
    print(y)

    print("---------IDFT-------------")
    x = idft2(y)
    print(x)

    print("---------ODFT-------------")
    dictshape = (4096, 64)
    OD = odftdict(dictshape=dictshape, isnorm=True)
    OD = odftndict(dictshape=dictshape, axis=2, isnorm=True)
    print(OD.shape)

    sio.savemat('OD.mat', {'OD': OD})

    plt.figure()
    plt.imshow(OD)
    plt.show()

    A = pys.showdict(OD, rcsize=(8, 8), stride=(0, 0), bgcolorv=-0.06)

    OD = pys.odftndict(dictshape=(4096, 64), isnorm=True)
    A1 = pys.showdict(OD, rcsize=(8, 8), stride=(0, 0), bgcolorv=-0.03)
