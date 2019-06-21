#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-06-13 10:34:43
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
import numpy as np


def mse(o, r):
    r"""Mean Squared Error

    The Mean Squared Error (MSE) is expressed as

    .. math::
        MSE = \frac{1}{MN}\sum_{i=1}^{M}\sum_{j=0}^{N}[{\bm I}(i,j), \hat{\bm I}(i, j)]^2

    Arguments
    ---------------
    o : ndarray
        Orignal signal matrix.

    r : ndarray 
        Reconstructed signal matrix

    Returns
    ---------------
    MSE : float
        Mean Squared Error

    """

    return np.mean((o.astype(float) - r.astype(float)) ** 2)


def rmse(o, r):
    r"""Root Mean Squared Error

    The Root Mean Squared Error (MSE) is expressed as

    .. math::
        RMSE = \sqrt{\frac{1}{MN}\sum_{i=1}^{M}\sum_{j=0}^{N}[{\bm I}(i,j), \hat{\bm I}(i, j)]^2}

    Arguments
    ---------------
    o : ndarray
        Orignal signal matrix.

    r : ndarray
        Reconstructed signal matrix

    Returns
    ---------------
    RMSE : float
        Root Mean Squared Error
    """

    return np.sqrt(np.mean((o.astype(float) - r.astype(float)) ** 2))


if __name__ == '__main__':

    o = np.array([[1, 2, 3], [4, 5, 6]])
    r = np.array([[0, 2, 3], [4, 5, 6]])

    print(mse(o, r))
    print(rmse(o, r))
