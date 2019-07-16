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

Fs = 256
Ts = 1
Ns = int(Ts * Fs)

f1 = 10
f2 = 20
f3 = 70

t = np.linspace(0, Ts, Ns)

y = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t) + \
    np.sin(2 * np.pi * f3 * t)

x = pys.idct1(y)
# x = pys.idct1(x)
f = np.linspace(0, Fs, Ns)


k1 = 2
k2 = 4
k3 = 6
k4 = 100


R = 4
alpha = 1e-6
M = np.size(y)
N = int(M * R)

ff = np.linspace(0, Fs, int(Ns * R))
A = pys.gaussian((M, N))
A = pys.odctdict((M, N), isnorm=True)
# A = pys.dctdict(N)
# A = pys.odctndict((M, N))
print(A.shape)

x1 = pys.romp(y, A, k=k1, alpha=alpha, verbose=False)
y1 = np.matmul(A, x1)

x2 = pys.romp(y, A, k=k2, alpha=alpha, verbose=False)
y2 = np.matmul(A, x2)

x3 = pys.romp(y, A, k=k3, alpha=alpha, verbose=False)
y3 = np.matmul(A, x3)

x4 = pys.romp(y, A, k=k4, alpha=alpha, verbose=False)
y4 = np.matmul(A, x4)

print("---MSE(y, y1) with k = " + str(k1) + ": ", pys.mse(y, y1))
print("---MSE(y, y2) with k = " + str(k2) + ": ", pys.mse(y, y2))
print("---MSE(y, y3) with k = " + str(k3) + ": ", pys.mse(y, y3))
print("---MSE(y, y4) with k = " + str(k4) + ": ", pys.mse(y, y4))

plt.figure()
plt.subplot(121)
plt.plot(t, y)
plt.xlabel('Time/s')
plt.ylabel('Amplitude')
plt.title('Orignal Signal (Time domain)')
plt.grid()

plt.subplot(122)
plt.plot(f, x)
# plt.plot(y)
plt.xlabel('Frequency/Hz')
plt.ylabel('Amplitude')
plt.title('Orignal Signal (frequency domain)')
plt.grid()
# plt.tight_layout()
plt.show()

plt.figure()
plt.subplot(221)
plt.plot(ff, x1)
plt.xlabel('Frequency/Hz')
plt.ylabel('Amplitude')
plt.title('Reconstructed Signal (k=' + str(k1) + ')')
plt.grid()

plt.subplot(222)
plt.plot(ff, x2)
plt.xlabel('Frequency/Hz')
plt.ylabel('Amplitude')
plt.title('Reconstructed Signal (k=' + str(k2) + ')')
plt.grid()

plt.subplot(223)
plt.plot(ff, x3)
plt.xlabel('Frequency/Hz')
plt.ylabel('Amplitude')
plt.title('Reconstructed Signal (k=' + str(k3) + ')')
plt.grid()

plt.subplot(224)
plt.plot(ff, x4)
plt.xlabel('Frequency/Hz')
plt.ylabel('Amplitude')
plt.title('Reconstructed Signal (k=' + str(k4) + ')')
plt.grid()
plt.tight_layout()
plt.show()


plt.figure()
plt.subplot(221)
plt.plot(t, y1)
plt.xlabel('Time/s')
plt.ylabel('Amplitude')
plt.title('Reconstructed Signal (k=' + str(k1) + ')')
plt.grid()

plt.subplot(222)
plt.plot(t, y2)
plt.xlabel('Time/s')
plt.ylabel('Amplitude')
plt.title('Reconstructed Signal (k=' + str(k2) + ')')
plt.grid()

plt.subplot(223)
plt.plot(t, y3)
plt.xlabel('Time/s')
plt.ylabel('Amplitude')
plt.title('Reconstructed Signal (k=' + str(k3) + ')')
plt.grid()

plt.subplot(224)
plt.plot(t, y4)
plt.xlabel('Time/s')
plt.ylabel('Amplitude')
plt.title('Reconstructed Signal (k=' + str(k4) + ')')
plt.grid()
plt.tight_layout()
plt.show()
