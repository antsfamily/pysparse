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
# dictype = 'DFT'
# dictype = None
mestype = 'Gaussian'
seed = 2019

if mestype is 'Gaussian':
    Phi = pys.gaussian((M, N), seed=seed, verbose=True)

print("===observation...")

if dictype is 'DCT':
    Psi = pys.dctdict(N)
    Psi = pys.dctmat(N)
if dictype is 'DFT':
    # Psi = pys.dctdict(N)
    Psi = pys.dftmat(N)

plt.figure()
plt.subplot(221)
plt.imshow(X)
plt.title('Orignal image signal')
if dictype is not None:
    plt.subplot(222)
    plt.imshow(np.abs(Psi))
    plt.title('Dictionary matrix (' + dictype + ')')
    plt.subplot(223)
    plt.imshow(np.abs(np.reshape(np.matmul(Psi, X), (N, N))))
    plt.title('Sparse Coefficient (' + dictype + ')')
    plt.colorbar()
plt.subplot(224)
plt.imshow(Phi)
plt.title('Measurement matrix (' + mestype + ')')
plt.tight_layout()
plt.show()

# ===========way 1=================
if dictype is not None:
    Z = np.matmul(Psi, X)
    Y = np.matmul(Phi, Z)
else:
    Y = np.matmul(Phi, X)

Z1 = pys.romp(
    Y, Phi, k=k1, alpha=alpha, normalize=True, tol=1e-16, verbose=True)
X1 = Z1
if dictype is 'DCT':
    X1 = np.matmul(Psi.transpose(), Z1)
if dictype is 'DFT':
    X1 = np.real(np.matmul(Psi.transpose(), Z1))

print("--MSE:", pys.mse(X, X1))

X1 = pys.scale(X1, sto=[0, 255], sfrom=sfrom, istrunc=True)
X1 = X1.astype(X.dtype)

PSNR1, MSE1, Vpeak1, dtype = pys.psnr(X, X1, Vpeak=None, mode='rich')
print("---PSNR1, MSE1, Vpeak1, dtype: ", PSNR1, MSE1, Vpeak1, dtype)

# ===========way 2=================

Y = np.matmul(Phi, X)
if dictype is not None:
    A = np.matmul(Phi, Psi.transpose())
else:
    A = Phi

Z2 = pys.romp(
    Y, A, k=k1, alpha=alpha, normalize=True, tol=1e-16, verbose=True)
X2 = Z2
if dictype is 'DCT':
    X2 = np.matmul(Psi.transpose(), Z2)
if dictype is 'DFT':
    X2 = np.real(np.matmul(Psi.transpose(), Z2))
print("--MSE:", pys.mse(X, X2))

X2 = pys.scale(X2, sto=[0, 255], sfrom=sfrom, istrunc=True)
X2 = X2.astype(X.dtype)

PSNR2, MSE2, Vpeak2, dtype = pys.psnr(X, X2, Vpeak=None, mode='rich')
print("---PSNR2, MSE2, Vpeak2, dtype: ", PSNR2, MSE2, Vpeak2, dtype)


plt.figure()
plt.subplot(121)
plt.imshow(X1)
plt.title(r'CS(${\bf y}={\bf \Phi}{\bf z},k=$' +
          str(k1) + ')' + '\nPSNR: %.2f' % PSNR1 + 'dB')
plt.subplot(122)
plt.imshow(X2)
plt.title(r'CS(${\bf y}={\bf \Phi}{\bf x},k=$' +
          str(k1) + ')' + '\nPSNR: %.2f' % PSNR2 + 'dB')
plt.tight_layout()
plt.show()
