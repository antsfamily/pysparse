#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-06-13 10:34:43
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$


import numpy as np


def normalization(x):
    x = x.astype('float32')
    mu = np.average(x)
    std = np.std(x)
    return (x - mu) / std, mu, std


def scale(X, sto=[0, 1], sfrom=None, istrunc=True, rich=False):
    r"""
    Scale data.

    .. math::
        x \in [a, b] --> y \in [c, d]

    .. math::
        y = (d-c)*(x-a) / (b-a) + c.

    Parameters
    ----------
    X : array_like
        The data to be scaled.
    sto : tuple, list, optional
        Specifies the range of data after beening scaled. Default [0, 1].
    sfrom : tuple, list, optional
        Specifies the range of data. Default [min(X), max(X)].
    istrunc : bool
        Specifies wether to truncate the data to [a, b], For example,
        If sfrom == [a, b] and 'istrunc' is true,
        then X[X < a] == a and X[X > b] == b.
    rich : bool
        If you want to see what the data is scaled from and scaled to,
        then you should set it to true
    Returns
    -------
    out : ndarray
        Scaled data array.
    sfrom, sto : list or tuple
        If rich is true, they will also be returned
    """

    if not(isinstance(sto, (tuple, list)) and len(sto) == 2):
        raise Exception("'sto' is a tuple or list, such as (-1,1)")
    if sfrom is not None:
        if not(isinstance(sfrom, (tuple, list)) and len(sfrom) == 2):
            raise Exception("'sfrom' is a tuple or list, such as (0, 255)")
    else:
        sfrom = [np.min(X) + 0.0, np.max(X) + 0.0]

    a = sfrom[0] + 0.0
    b = sfrom[1] + 0.0
    c = sto[0] + 0.0
    d = sto[1] + 0.0

    X = X.astype('float')

    if istrunc:
        X[X < a] = a
        X[X > b] = b

    if rich:
        return (X - a) * (d - c) / (b - a) + c, sfrom, sto
    else:
        return (X - a) * (d - c) / (b - a) + c
