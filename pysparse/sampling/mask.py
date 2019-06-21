#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-07-06 10:38:13
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
#
# @Note    :
#
import numpy as np
from pysparse.utils.const import *
import scipy.io as sio


def linemask(L, shape, verbose=False):
    r"""The indicator of the domain in 2D fourier space for the specified line geometry.

    Returns the indicator of the domain in 2D fourier space for the specified line geometry.

    Arguments
    --------------
    L : integer
        number of lines
    shape : tuple
        mask shape
    """

    print("================in linemask================")
    if verbose:
        print("===generates line mask...")

    thc = np.linspace(0, np.pi - np.pi / L, L, endpoint=True)

    X = np.zeros(shape)
    N = shape[0]
    for ll in range(L):
        if (thc[ll] <= np.pi / 4.0) or (thc[ll] > 3.0 * np.pi / 4.0):
            yr = np.round(
                np.tan(thc[ll]) *
                np.linspace(-N / 2 + 1, N / 2 - 1, N - 1, endpoint=True)) +\
                N / 2
            yr = yr.astype('int')

            for nn in range(N - 2):
                X[yr[nn], nn + 1] = 1
        else:
            xc = np.round(
                (1.0 / np.tan(thc[ll])) *
                np.linspace(-N / 2 + 1, N / 2 - 1, N - 1, endpoint=True)) +\
                N / 2
            xc = xc.astype('int')

            for nn in range(N - 2):
                X[nn + 1, xc[nn]] = 1
    if verbose:
        print("===Done!")
    return X


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pysparse as pys

    L = 8
    shape = (128, 128)
    X = pys.linemask(L=L, shape=shape)
    sio.savemat('mask.mat', {'X': X})

    plt.figure()
    plt.imshow(X)
    plt.show()
