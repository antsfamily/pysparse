#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-07-06 10:38:13
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
#
#

import pysparse as pys
import numpy as np
from scipy.misc import imread
from skimage import transform
import matplotlib.pyplot as plt


imgfile = '../../data/img/cameraman.tif'
# imgfile = '../../data/img/tomography.bmp'
# imgfile = '../../data/img/lena.bmp'

X = imread(imgfile)
print(X.shape)
# X = transform.resize(X, (64, 64))
# X = X / 255.0

H, W = X.shape


sfrom = [0, 255]
# sfrom = None
sto = [0, 255]

alpha = 0.000001
CR = 4
N = H
M = int(N / CR)
k1 = int(N / 8)

dictype = 'DCT'
mestype = 'Gaussian'
seed = 2019

if mestype is 'Gaussian':
    Phi = pys.gaussian((M, N), seed=seed, verbose=True)

print("===observation...")

if dictype is 'DCT':
    D = pys.dctdict(N)

A = np.matmul(Phi, D)

plt.figure()
plt.subplot(221)
plt.imshow(X)
plt.title('Orignal image signal')
plt.subplot(222)
plt.imshow(D)
plt.title('Dictionary matrix (' + dictype + ')')
plt.subplot(223)
plt.imshow(np.reshape(np.matmul(D, X), (N, N)))
plt.title('Sparse Coefficient (' + dictype + ')')
plt.colorbar()
plt.subplot(224)
plt.imshow(Phi)
plt.title('Measurement matrix (' + mestype + ')')
plt.tight_layout()
plt.show()

# ===========way 1=================

Y = np.matmul(Phi, X)

X1 = pys.romp(
    Y, Phi, k=k1, alpha=alpha, normalize=True, tol=1e-16, verbose=True)
if dictype is not None:
    X1 = np.matmul(D, X1)
X1 = pys.scale(X1, sto=[0, 255], sfrom=sfrom, istrunc=True)
X1 = X1.astype(X.dtype)

PSNR1, MSE1, Vpeak1, dtype = pys.psnr(X, X1, Vpeak=None, mode='rich')
print("---PSNR1, MSE1, Vpeak1, dtype: ", PSNR1, MSE1, Vpeak1, dtype)

# ===========way 2=================

Y = np.matmul(Phi, X)

X2 = pys.romp(
    Y, A, k=k1, alpha=alpha, normalize=True, tol=1e-16, verbose=True)
if dictype is not None:
    X2 = np.matmul(D, X2)
X2 = pys.scale(X2, sto=[0, 255], sfrom=sfrom, istrunc=True)
X2 = X2.astype(X.dtype)

PSNR2, MSE2, Vpeak2, dtype = pys.psnr(X, X2, Vpeak=None, mode='rich')
print("---PSNR2, MSE2, Vpeak2, dtype: ", PSNR2, MSE2, Vpeak2, dtype)


# ===========way 3=================

Y = np.matmul(A, X)

X3 = pys.romp(
    Y, Phi, k=k1, alpha=alpha, normalize=True, tol=1e-16, verbose=True)
if dictype is not None:
    X3 = np.matmul(D, X3)
X3 = pys.scale(X3, sto=[0, 255], sfrom=sfrom, istrunc=True)
X3 = X3.astype(X.dtype)

PSNR3, MSE3, Vpeak3, dtype = pys.psnr(X, X3, Vpeak=None, mode='rich')
print("---PSNR3, MSE3, Vpeak3, dtype: ", PSNR3, MSE3, Vpeak3, dtype)

# ===========way 4=================

Y = np.matmul(A, X)

X4 = pys.romp(
    Y, A, k=k1, alpha=alpha, normalize=True, tol=1e-16, verbose=True)
if dictype is not None:
    X4 = np.matmul(D, X4)
X4 = pys.scale(X4, sto=[0, 255], sfrom=sfrom, istrunc=True)
X4 = X4.astype(X.dtype)

PSNR4, MSE4, Vpeak4, dtype = pys.psnr(X, X4, Vpeak=None, mode='rich')
print("---PSNR6, MSE6, Vpeak6, dtype: ", PSNR4, MSE4, Vpeak4, dtype)

plt.figure()
plt.subplot(221)
plt.imshow(X1)
plt.title(r'CS(${\bf \Phi},{\bf \Phi},k=$' +
          str(k1) + ')' + '\nPSNR: %.2f' % PSNR1 + 'dB')
plt.subplot(222)
plt.imshow(X2)
plt.title(r'CS(${\bf \Phi},{\bf A},k=$' +
          str(k1) + ')' + '\nPSNR: %.2f' % PSNR2 + 'dB')
plt.subplot(223)
plt.imshow(X3)
plt.title(r'CS(${\bf A},{\bf \Phi},k=$' +
          str(k1) + ')' + '\nPSNR: %.2f' % PSNR3 + 'dB')
plt.subplot(224)
plt.imshow(X4)
plt.title(r'CS(${\bf A},{\bf A},k=$' +
          str(k1) + ')' + '\nPSNR: %.2f' % PSNR4 + 'dB')
plt.tight_layout()
plt.show()
