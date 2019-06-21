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


imgfile = '../../data/img/tomography.bmp'
imgfile = '../../data/img/lena.bmp'

X = imread(imgfile)
print(X.shape)
# X = transform.resize(X, (64, 64))
# X = X / 255.0


H, W = X.shape

sfrom = [0, 255]
# sfrom = None
sto = [0, 255]

alpha = 0.000001
CR = 16
N = H
M = int(N / CR)
k1 = int(N / 32)
k2 = int(N / 16)
k3 = int(N / 8)
k4 = int(N / 4)
k5 = int(N / 2)
k6 = int(N / 1)

A = pys.gaussian((M, N))
A = pys.odctdict((M, N))
# A = pys.odctndict((M, N))
print(A.shape)

Y = np.matmul(A, X)

plt.figure()
plt.subplot(121)
plt.imshow(X)
plt.title('Orignal Image (X)')
plt.subplot(122)
plt.imshow(Y)
plt.title('Signal Image (Y=AX)')
plt.tight_layout()
plt.show()

X1 = pys.romp(
    Y, A, k=k1, alpha=alpha, normalize=True, tol=1e-16, verbose=True)

# X1 = np.matmul(D, X1)
X1 = pys.scale(X1, sto=[0, 255], sfrom=sfrom, istrunc=True)
X1 = X1.astype(X.dtype)


PSNR1, MSE1, Vpeak1, dtype = pys.psnr(X, X1, Vpeak=None, mode='rich')


X2 = pys.romp(
    Y, A, k=k2, alpha=alpha, normalize=True, tol=1e-16, verbose=True)

# X2 = np.matmul(D, X2)
X2 = pys.scale(X2, sto=[0, 255], sfrom=sfrom, istrunc=True)
X2 = X2.astype(X.dtype)

PSNR2, MSE2, Vpeak2, dtype = pys.psnr(X, X2, Vpeak=None, mode='rich')

X3 = pys.romp(
    Y, A, k=k3, alpha=alpha, normalize=True, tol=1e-16, verbose=True)

# X3 = np.matmul(D, X3)
X3 = pys.scale(X3, sto=[0, 255], sfrom=sfrom, istrunc=True)
X3 = X3.astype(X.dtype)

PSNR3, MSE3, Vpeak3, dtype = pys.psnr(X, X3, Vpeak=None, mode='rich')

X4 = pys.romp(
    Y, A, k=k4, alpha=alpha, normalize=True, tol=1e-16, verbose=True)

# X4 = np.matmul(D, X4)
X4 = pys.scale(X4, sto=[0, 255], sfrom=sfrom, istrunc=True)
X4 = X4.astype(X.dtype)

PSNR4, MSE4, Vpeak4, dtype = pys.psnr(X, X4, Vpeak=None, mode='rich')

X5 = pys.romp(
    Y, A, k=k5, alpha=alpha, normalize=True, tol=1e-16, verbose=True)

# X5 = np.matmul(D, X5)
X5 = pys.scale(X5, sto=[0, 255], sfrom=sfrom, istrunc=True)
X5 = X5.astype(X.dtype)

PSNR5, MSE5, Vpeak5, dtype = pys.psnr(X, X5, Vpeak=None, mode='rich')

X6 = pys.romp(
    Y, A, k=k6, alpha=alpha, normalize=True, tol=1e-16, verbose=True)

# X6 = np.matmul(D, X6)
X6 = pys.scale(X6, sto=[0, 255], sfrom=sfrom, istrunc=True)
X6 = X6.astype(X.dtype)

PSNR6, MSE6, Vpeak6, dtype = pys.psnr(X, X6, Vpeak=None, mode='rich')


print("---PSNR1, MSE1, Vpeak1, dtype: ", PSNR1, MSE1, Vpeak1, dtype)
print("---PSNR2, MSE2, Vpeak2, dtype: ", PSNR2, MSE2, Vpeak2, dtype)
print("---PSNR3, MSE3, Vpeak3, dtype: ", PSNR3, MSE3, Vpeak3, dtype)
print("---PSNR4, MSE4, Vpeak4, dtype: ", PSNR4, MSE4, Vpeak4, dtype)
print("---PSNR5, MSE5, Vpeak5, dtype: ", PSNR5, MSE5, Vpeak5, dtype)
print("---PSNR6, MSE6, Vpeak6, dtype: ", PSNR6, MSE6, Vpeak6, dtype)

plt.figure()
plt.subplot(321)
plt.imshow(X1)
plt.title('OMP(k=' + str(k1) + '),' + '\nPSNR: %.2f' % PSNR1 + 'dB')
plt.subplot(322)
plt.imshow(X2)
plt.title('OMP(k=' + str(k2) + '),' + '\nPSNR: %.2f' % PSNR2 + 'dB')
plt.subplot(323)
plt.imshow(X3)
plt.title('OMP(k=' + str(k3) + '),' + '\nPSNR: %.2f' % PSNR3 + 'dB')
plt.subplot(324)
plt.imshow(X4)
plt.title('OMP(k=' + str(k4) + '),' + '\nPSNR: %.2f' % PSNR4 + 'dB')
plt.subplot(325)
plt.imshow(X5)
plt.title('OMP(k=' + str(k5) + '),' + '\nPSNR: %.2f' % PSNR5 + 'dB')
plt.subplot(326)
plt.imshow(X6)
plt.title('OMP(k=' + str(k6) + '),' + '\nPSNR: %.2f' % PSNR6 + 'dB')
plt.tight_layout()
plt.show()
