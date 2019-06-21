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

from sklearn.linear_model import OrthogonalMatchingPursuit


def omp0(y, A, normalize=False, tol=1.0e-6, verbose=False):
    r"""omp

    The optimization objective for Lasso is::
    .. math::
        (1 / (2 * n_samples)) * ||y - Xw||^2_2 + ||w||_0

    Technically the Lasso model is optimizing the same objective function as
    the Elastic Net with ``l1_ratio=1.0`` (no L2 penalty).

    Arguments
     y {[type]} -- [description]
     A {[type]} -- [description]

    Keyword Arguments:
     alpha {float, optional} -- Constant that multiplies the L1 term. (default: {0.5})
     normalize {boolean} -- If True, the regressors X will be normalized before regression by
                            subtracting the mean and dividing by the l2-norm. (default: {True})
     max_iter {int} -- The maximum number of iterations (default: {200})
     tol {float} -- The tolerance for the optimization (default: {1.0e-6})
    """

    if verbose:
        print("================in omp================")
        print("===Do OMP...")
    rgr_omp = OrthogonalMatchingPursuit(normalize=normalize, tol=tol)
    rgr_omp.fit(A, y)
    x = rgr_omp.coef_
    if verbose:
        print("===Done!")

    return x


def omp(Y, A, k=None, normalize=False, tol=1.0e-6, verbose=False):
    r"""Orthogonal Matching Pursuit

    OMP find the sparse most decomposition 

    .. math::
       {\bm y} = {\bm A}{\bm \alpha} = {\alpha}_1 {\bm a}_1 + {\alpha}_2 {\bm a}_2 + \cdots {\alpha}_n {\bm a}_n,
       :label: equ-OmpProb

    The optimization objective for omp is::

    .. math::
       {\rm min}_{\bm{x}} = \|{\bm x}\|_p + \lambda\|{\bm y} - {\bm c}{\bm x}\|_2.
       :label: equ-CS1d_Optimizationnost

    Parameters
    --------------
    y : ndarray
        signal vector or matrix, if :math:`{\bm y}\in{\mathbb R}^{M\times 1}` is a matrix, 
        then apply OMP on each column 
    A : ndarary
        overcomplete dictionary (:math:`{\bm A}\in {\mathbb R}^{M\times N}` )

    Keyword Arguments
    ------------------
    k : integer
        The sparse degree (default: size of :math:`{\bm x}`)

    normalize : boolean
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm. (default: {True})

    tol : float
        The tolerance for the optimization (default: {1.0e-6})

    verbose : boolean
        show more log info.
    """

    if verbose:
        print("================in omp================")
        print("===Do OMP...")

    M, N = A.shape

    if k is None:
        k = int(N / 4)

    vecflag = False
    if np.ndim(Y) < 2:
        vecflag = True
        Y = np.reshape(Y, (np.size(Y), 1))

    MY, NY = Y.shape

    X = np.zeros((N, NY))

    for n in range(NY):
        # print(n)
        y = Y[:, n]
        r = y
        I = []

        for t in range(k):
            it = np.argmax(np.abs(np.dot(r, A)))
            I.append(it)
            AI = A[:, I]
            s = np.matmul(
                np.matmul(np.linalg.inv(np.matmul(AI.transpose(), AI)),
                          AI.transpose()), y)

            # s = np.matmul(AI.transpose(), y)
            yHat = np.matmul(AI, s)
            r = y - yHat

        x = np.zeros(N)
        x[I] = s
        X[:, n] = x

    if vecflag:
        X = np.reshape(X, N)

    if verbose:
        print("===Done!")

    return X


def romp(Y, A, k=None, alpha=1e-6, normalize=False, tol=1.0e-6, verbose=False):
    r"""Regularized Orthogonal Matching Pursuit

    ROMP add a small penalty factor :math:`\alpha` to

    .. math::
        ({\bm A}_{{\mathbb I}_t}^T{\bm A}_{{\mathbb I}_t})^{-1}`

    to avoid matrix singularity

    .. math::
        ({\bm A}_{{\mathbb I}_t}^T{\bm A}_{{\mathbb I}_t} + \alpha {\bm I})^{-1}

    where, :math:`\alpha > 0`. 

    Parameters
    --------------
    y : ndarray
        signal vector or matrix, if :math:`{\bm y}\in{\mathbb R}^{M\times 1}` is a matrix, 
        then apply OMP on each column 

    A : ndarary
        overcomplete dictionary ( :math:`{\bm A}\in {\mathbb R}^{M\times N}` )

    Keyword Arguments
    ----------------------
    k : integer
        The sparse degree (default: size of :math:`{\bm x}`)

    alpha : float 
        The regularization factor (default: 1.0e-6)

    normalize : boolean
        If True, Y will be normalized before decomposition by subtracting the mean and dividing by the l2-norm. (default: {True})

    tol : float
        The tolerance for the optimization (default: {1.0e-6})

    verbose : boolean
        show more log info.
    """

    if verbose:
        print("================in omp================")
        print("===Do OMP...")

    M, N = A.shape

    if k is None:
        k = N

    vecflag = False
    if np.ndim(Y) < 2:
        vecflag = True
        Y = np.reshape(Y, (np.size(Y), 1))

    MY, NY = Y.shape

    X = np.zeros((N, NY))

    for n in range(NY):
        # print(n)
        y = Y[:, n]
        r = y
        I = []

        for t in range(k):
            it = np.argmax(np.abs(np.dot(r, A)))
            I.append(it)
            AI = A[:, I]
            MAI, NAI = AI.shape
            E = alpha * np.eye(NAI, NAI)
            s = np.matmul(
                np.matmul(np.linalg.inv(np.matmul(AI.transpose(), AI) + E),
                          AI.transpose()), y)

            # s = np.matmul(AI.transpose(), y)
            yHat = np.matmul(AI, s)
            r = y - yHat

        x = np.zeros(N)
        x[I] = s
        X[:, n] = x

    if vecflag:
        X = np.reshape(X, N)

    if verbose:
        print("===Done!")

    return X


if __name__ == '__main__':

    import pysparse as pys

    M = 3
    N = 4
    x = np.array([-0.5, 0.01, 3.1, 0.8])
    A = pys.gaussian((M, N))
    # A = pys.odctdict((M, N))

    y = np.matmul(A, x)

    print("---A (Mesurement Matrix): ", A)
    print("---y (Mesurement Vector): ", y)
    print("---x (Orignal Signal): ", x)
    x = pys.romp(y, A, k=2, alpha=0.0001, verbose=False)
    print("---x (Reconstructed Signal): ", x)
    x = pys.omp(y, A, k=3, verbose=False)
    print("---x (Reconstructed Signal): ", x)
