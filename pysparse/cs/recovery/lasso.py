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

from sklearn.linear_model import Lasso


def lasso(y, A, alpha=0.5, normalize=False, max_iter=200, tol=1.0e-6, verbose=False):
    r"""lasso

    The optimization objective for Lasso is::

    .. math::
        (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    Technically the Lasso model is optimizing the same objective function as
    the Elastic Net with ``l1_ratio=1.0`` (no L2 penalty).

    Arguments:
     y {[type]} -- [description]
     A {[type]} -- [description]

    Keyword Arguments:
     alpha {float, optional} -- Constant that multiplies the L1 term. (default: {0.5})
     normalize {boolean} -- If True, the regressors X will be normalized before regression by
                            subtracting the mean and dividing by the l2-norm. (default: {True})
     max_iter {int} -- The maximum number of iterations (default: {200})
     tol {float} -- The tolerance for the optimization (default: {1.0e-6})
    """

    print(max_iter, tol)

    if verbose:
        print("================in lasso================")
        print("===Do Lasso L1...")
    rgr_lasso = Lasso(
        alpha=alpha, normalize=normalize, max_iter=max_iter, tol=tol)
    rgr_lasso.fit(A, y)
    x = rgr_lasso.coef_
    if verbose:
        print("===Done!")

    return x
