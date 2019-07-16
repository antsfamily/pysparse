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

D = pys.dctmat(H)

print("DCT IDCT error:", np.mean(np.abs(pys.idct1(Yc, axis=0) - X)))
print("DCT IDCT error:", np.mean(np.abs(pys.idct1(Yr, axis=1) - X)))

plt.figure()
plt.subplot(221)
plt.imshow(X)
plt.title('Orignal Signal')
plt.subplot(222)
plt.imshow(np.log(np.abs(Yc)))
plt.title('DCT-1Dc Coefficients')
plt.subplot(223)
plt.imshow(np.log(np.abs(Yr)))
plt.title('DCT-1Dr Coefficients')
plt.subplot(224)
plt.imshow(D)
plt.title('DCT-1D Matrix')
plt.tight_layout()
plt.show()
