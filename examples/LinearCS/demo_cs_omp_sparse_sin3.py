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

Fs = 1936
Ts = 1
Ns = int(Ts * Fs)

f1 = 100
f2 = 200
f3 = 700

t = np.linspace(0, Ts, Ns)

xo = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t) + \
    np.sin(2 * np.pi * f3 * t)

x = np.abs(np.fft.fftshift(np.fft.fft(xo)))
f = np.linspace(-Fs / 2, Fs / 2, Ns)

CR = 2

k1 = 2
k2 = 4
k3 = 6
k4 = 100

alpha = 0.0000001
N = np.size(x)
M = int(N / CR)
A = pys.gaussian((M, N))
A = pys.odctdict((M, N))
# A = pys.odctndict((M, N))
print(A.shape)
y = np.matmul(A, x)

x1 = pys.romp(y, A, k=k1, alpha=alpha, verbose=False)

x2 = pys.romp(y, A, k=k2, alpha=alpha, verbose=False)

x3 = pys.romp(y, A, k=k3, alpha=alpha, verbose=False)

x4 = pys.romp(y, A, k=k4, alpha=alpha, verbose=False)

print("---MSE(x, x1) with k = " + str(k1) + ": ", pys.mse(x, x1))
print("---MSE(x, x2) with k = " + str(k2) + ": ", pys.mse(x, x2))
print("---MSE(x, x3) with k = " + str(k3) + ": ", pys.mse(x, x3))
print("---MSE(x, x4) with k = " + str(k4) + ": ", pys.mse(x, x4))

plt.figure()
plt.subplot(121)
plt.plot(t, xo)
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
plt.tight_layout()
plt.show()

plt.figure()
plt.subplot(221)
plt.plot(f, x1)
plt.xlabel('Frequency/Hz')
plt.ylabel('Amplitude')
plt.title('Reconstructed Signal (k=' + str(k1) + ')')
plt.grid()

plt.subplot(222)
plt.plot(f, x2)
plt.xlabel('Frequency/Hz')
plt.ylabel('Amplitude')
plt.title('Reconstructed Signal (k=' + str(k2) + ')')
plt.grid()

plt.subplot(223)
plt.plot(f, x3)
plt.xlabel('Frequency/Hz')
plt.ylabel('Amplitude')
plt.title('Reconstructed Signal (k=' + str(k3) + ')')
plt.grid()

plt.subplot(224)
plt.plot(f, x4)
plt.xlabel('Frequency/Hz')
plt.ylabel('Amplitude')
plt.title('Reconstructed Signal (k=' + str(k4) + ')')
plt.grid()
plt.tight_layout()
plt.show()
