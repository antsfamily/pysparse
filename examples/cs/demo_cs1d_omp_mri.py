#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-07-06 10:38:13
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
#
#

import numpy as np
import pysparse as pys
import scipy.io as sio
import matplotlib.pyplot as plt


# https://github.com/leoliuf/MRiLab/blob/master/Resources/VObj/BrainHighResolution.mat
# datafile = '../../data/mri/BrainHighResolution.mat'
datafile = '../../data/mri/MatlabPhantom.mat'

data = sio.loadmat(datafile)

I = data['img']
H, W = I.shape
I = I * 255
I = I.astype('uint8')
# generate raw data
X = np.fft.fft2(I)
# X = X.flatten()

sfrom = [0, 255]
# sfrom = None
sto = [0, 255]

CR = 4
dictype = 'DCT'
# dictype = None
mestype = 'Gaussian'

M = int(H * W / CR)
N = int(H * W)
k = int(H * W)

M = int(H / CR)
N = int(H)
k = 100

if mestype is 'Gaussian':
    Phi = pys.gaussian((M, N), verbose=True)

if dictype is 'DCT':
    Psi = pys.dctdict(N)
else:
    Psi = None

print("===observation...")

y = np.matmul(Phi, X)
print(Phi.shape, X.shape, y.shape)

X1 = pys.cs1d(y, Phi, Psi=None, optim='OMP', k=k,
              tol=None, osshape=(H, W), verbose=True)
I1 = np.fft.ifft2(X1)
I1 = pys.scale(I1, sto=[0, 255], sfrom=sfrom, istrunc=True)
I1 = I1.astype('uint8')

PSNR1, MSE1, Vpeak1, dtype = pys.psnr(I, I1, Vpeak=None, mode='rich')


print("---PSNR1, MSE1, Vpeak1, dtype: ", PSNR1, MSE1, Vpeak1, dtype)

plt.figure()
plt.subplot(221)
plt.imshow(I)
plt.colorbar()
plt.title('Image')
plt.subplot(222)
plt.imshow(np.log10(np.abs(X)))
plt.colorbar()
plt.title('Amplitude')
plt.subplot(223)
plt.imshow(np.angle(X))
plt.colorbar()
plt.title('Phase')
plt.subplot(224)
plt.imshow(I1)
plt.colorbar()
plt.title('CS OMP(k=' + str(k) + '),' + '\nPSNR: %.2f' % PSNR1 + 'dB')
plt.tight_layout()
plt.show()
