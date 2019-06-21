#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-07-06 10:38:13
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
#

import numpy as np
import matplotlib.pyplot as plt
import pysparse as pys
import scipy.io as sio


print("----------DCT-1D------------")
N1 = 64
rcsize1 = (int(np.sqrt(N1)), int(np.sqrt(N1)))
OD = pys.dctdict(N1)
A1 = pys.showdict(OD, rcsize=rcsize1, stride=(0, 0), plot=False)

N2 = 256
rcsize2 = (int(np.sqrt(N2)), int(np.sqrt(N2)))
OD = pys.dctdict(N2)
A2 = pys.showdict(OD, rcsize=rcsize2, stride=(0, 0), plot=False)

N3 = 1024
rcsize3 = (int(np.sqrt(N3)), int(np.sqrt(N3)))
OD = pys.dctdict(N3)
A3 = pys.showdict(OD, rcsize=rcsize3, stride=(0, 0), plot=False)

N4 = 4096
rcsize4 = (int(np.sqrt(N4)), int(np.sqrt(N4)))
OD = pys.dctdict(N4)
A4 = pys.showdict(OD, rcsize=rcsize4, stride=(0, 0), plot=False)

plt.figure()
plt.subplot(221)
plt.imshow(A1)
plt.colorbar()
plt.title('DCT: ' + str((N1, N1)))
plt.subplot(222)
plt.imshow(A2)
plt.colorbar()
plt.title('DCT: ' + str((N2, N2)))
plt.subplot(223)
plt.imshow(A3)
plt.colorbar()
plt.title('DCT: ' + str((N3, N3)))
plt.subplot(224)
plt.imshow(A4)
plt.colorbar()
plt.title('DCT: ' + str((N4, N4)))
plt.tight_layout()
plt.show()
