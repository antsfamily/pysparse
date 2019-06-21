#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-07-06 10:38:13
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
#

import numpy as np
import pysparse as pys
import matplotlib.pyplot as plt
from scipy.misc import imread

imgfile = '../../../data/img/lena.bmp'

X = imread(imgfile)
print(X.shape)

H, W = X.shape


Yc = pys.dct1(X, axis=0)

Yr = pys.dct1(X, axis=1)

Y = pys.dct2(X)
XX = pys.idct2(Y)

D = pys.dctmat(H)
D = D.transpose()

plt.figure()
plt.subplot(221)
plt.imshow(X)
plt.title('Orignal Signal')
plt.subplot(222)
plt.imshow(np.log(np.abs(Y)))
plt.title('DCT-2D Coefficients')
plt.subplot(223)
plt.imshow(XX)
plt.title('IDCT-2D Reconstruction')
plt.subplot(224)
plt.imshow(D)
plt.title('IDCT-1D Matrix')
plt.tight_layout()
plt.show()
