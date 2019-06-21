#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-06-13 10:34:43
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

from pysparse.evaluation.error import mse

import numpy as np


def snr():
    pass


def psnr(o, r, Vpeak=None, mode='simple'):
    r"""Peak Signal-to-Noise Ratio

    The Peak Signal-to-Noise Ratio (PSNR) is expressed as

    .. math::
        10 \log10(\frac{V_{peak}^2}{\rm MSE})

    For float data, V_{peak} = 1;

    For interges, :math:`V_{peak} = 2^{nbits}`,
    e.g. uint8: 255, uint16: 65535 ...

    Parameters
    -----------
    o : array_like
        Reference data array. For image, it's the original image.
    r : array_like
        The data to be compared. For image, it's the reconstructed image.
    Vpeak : float, int or None, optional
        The peak value. If None, computes automaticly.
    mode : str or None, optional
         'simple' or 'rich'. 'simple' (default) --> just return psnr i.e.
         'rich' --> return psnr, mse, Vpeak, imgtype.

    Returns
    -------
    PSNR : float
        Peak Signal to Noise Ratio value.

    """

    if o.dtype != r.dtype:
        print("Warning: o(" + str(o.dtype) + ")and r(" + str(r.dtype) +
              ")have different type! PSNR may not right!")

    if Vpeak is None:
        if o.dtype in ('float', 'float16', 'float32', 'float64'):
            Vpeak = 1
        elif o.dtype in ('uint8', 'uint16', 'uint32', 'uint64'):
            datatype = str(o.dtype)
            Vpeak = 2 ** int(datatype[4:]) - 1
        elif o.dtype in ('int64', 'int32', 'int16', 'int8'):
            datatype = str(o.dtype)
            Vpeak = 2 ** int(datatype[3:]) / 2 - 1
        else:
            raise TypeError('Unrecognized type!')

    MSE = mse(o, r)
    PSNR = 10 * np.log10((Vpeak ** 2) / MSE)
    if mode is None:
        mode = 'simple'
    if mode == 'rich':
        return PSNR, MSE, Vpeak, o.dtype
    else:
        return PSNR


if __name__ == '__main__':
    import pysparse as pys

    o = np.array([[251, 200, 210], [220, 5, 6]])
    r = np.array([[0, 200, 210], [220, 5, 6]])
    PSNR, MSE, Vpeak, dtype = pys.psnr(o, r, Vpeak=None, mode='rich')
    print(PSNR, MSE, Vpeak, dtype)

    o = np.array([[251, 200, 210], [220, 5, 6]]).astype('uint8')
    r = np.array([[0, 200, 210], [220, 5, 6]]).astype('uint8')
    PSNR, MSE, Vpeak, dtype = pys.psnr(o, r, Vpeak=None, mode='rich')
    print(PSNR, MSE, Vpeak, dtype)

    o = np.array([[251, 200, 210], [220, 5, 6]]).astype('float')
    r = np.array([[0, 200, 210], [220, 5, 6]]).astype('float')
    PSNR, MSE, Vpeak, dtype = pys.psnr(o, r, Vpeak=None, mode='rich')
    print(PSNR, MSE, Vpeak, dtype)
