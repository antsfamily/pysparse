#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-06-13 10:34:43
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
import numpy as np
from pysparse.utils.const import *
import matplotlib.pyplot as plt


def showdict(D, rcsize=None, stride=None, plot=True, bgcolorv=0, cmap=None, title=None, xlabel=None, ylabel=None):
    r"""
    Trys to show image blocks in one image.

    Parameters
    ----------
    D : array_like
        Blocks to be shown, a bH-bW-bC-bN numpy ndarray.
    rcsize : int tuple or None, optional
        Specifies how many rows and cols of blocks that you want to show,
        e.g. (rows, cols). If not given, rcsize=(rows, clos) will be computed
        automaticly.
    stride : int tuple or None, optional
        The step size (blank pixels nums) in row and col between two blocks.
        If not given, stride=(1,1).
    plot : bool, optional
        True for ploting, False for silent and returns a H-W-C numpy ndarray
        for showing.
    bgcolorv : float or None, optional
        The background color, 1 for white, 0 for black. Default, 0.

    Returns
    -------
    out : ndarray or bool
        A H-W-C numpy ndarray for showing.

    See Also
    --------
    odctdict.

    Examples
    --------
    >>> D = pys.odctdict((M, N))
    >>> showdict(D, bgcolor='k')

    """

    # M, N = D.shape
    # print(M, N)
    # H1 = int(np.sqrt(M))
    # W1 = int(np.sqrt(M))
    # D = np.reshape(D, (H1, W1, 1, N))

    # H = int(np.sqrt(N))
    # W = int(np.sqrt(N))

    # A = np.zeros((int(H * H1), int(W * W1)))

    # rows, cols = np.mgrid[0:H1, 0:W1]
    # print(rows)
    # for j in range(W):
    #     for i in range(H):

    #         A[int(i * H1) + rows, int(j * W1) +
    #           cols] = D[:, :, 0, int(i + j * H)]

    # plt.figure()
    # plt.imshow(A)
    # plt.show()

    if plot is None:
        plot = True

    M, N = D.shape
    H1 = int(np.sqrt(M))
    W1 = int(np.sqrt(M))
    D = np.reshape(D, (H1, W1, 1, N))

    if D.size == 0:
        return D
    if not (isinstance(D, np.ndarray) and D.ndim == 4):
        raise TypeError('"D" should be a pH-pW-pC-pN numpy array!')

    _, _, _, bN = D.shape

    if rcsize is None:
        rows = int(math.sqrt(bN))
        cols = int(bN / rows)
        if bN % cols > 0:
            rows = rows + 1
    else:
        rows = rcsize[0]
        cols = rcsize[1]
    # step size
    if stride is None:
        stride = (1, 1)
    # background color
    # if bgcolor == 'w':
    #     bgcolor_value = 255
    # elif bgcolor == 'k':
    #     bgcolor_value = 0
    bgcolor_value = bgcolorv
    if bN < rows * cols:
        A = np.pad(D,
                   ((0, stride[0]), (0, stride[1]),
                       (0, 0), (0, rows * cols - bN)),
                   'constant', constant_values=bgcolor_value)
    else:
        A = np.pad(D,
                   ((0, stride[0]), (0, stride[1]), (0, 0), (0, 0)),
                   'constant', constant_values=bgcolor_value)
        A = A[:, :, :, 0:rows * cols]

    aH, aW, aC, aN = A.shape

    A = np.transpose(A, (3, 1, 0, 2)).reshape(
        rows, cols, aH, aW, aC).swapaxes(
        1, 2).reshape(rows * aH, cols * aW, aC)
    # A = np.transpose(A, (3, 0, 1, 2)).reshape(
    #     rows, cols, aH, aW, aC).swapaxes(
    #     1, 2).reshape(rows * aH, cols * aW, aC)

    aH, aW, aC = A.shape
    A = A[0:aH - stride[0], 0:aW - stride[1], :]
    if aC == 1:
        A = A[:, :, 0]

    if title is None:
        title = 'Show ' + str(bN) + ' atoms in ' + str(rows) +\
            ' rows ' + str(cols) + ' cols, with stride' + str(stride)
    if xlabel is None:
        xlabel = ''
    if ylabel is None:
        ylabel = ''
    if plot:
        plt.figure()
        plt.imshow(A, cmap=cmap)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.colorbar()
        plt.show()

    return A    # H-W-C
