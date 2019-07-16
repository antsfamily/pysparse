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


print("----------ODCT-1D------------")
ds1 = (16, 8)
rs, cs = (int(np.sqrt(ds1[1])), int(np.sqrt(ds1[1])))
OD = pys.odctdict(dictshape=ds1, isnorm=True)
A1 = pys.showdict(OD, rcsize=(rs, cs), stride=(0, 0), plot=False)

ds2 = (16, 16)
rs, cs = (int(np.sqrt(ds2[1])), int(np.sqrt(ds2[1])))
OD = pys.odctdict(dictshape=ds2, isnorm=True)
A2 = pys.showdict(OD, rcsize=(rs, cs), stride=(0, 0), plot=False)

ds3 = (16, 32)
rs, cs = (int(np.sqrt(ds3[1])), int(np.sqrt(ds3[1])))
OD = pys.odctdict(dictshape=ds3, isnorm=True)
A3 = pys.showdict(OD, rcsize=(rs, cs), stride=(0, 0), plot=False)

ds4 = (16, 64)
rs, cs = (int(np.sqrt(ds4[1])), int(np.sqrt(ds4[1])))
OD = pys.odctdict(dictshape=ds4, isnorm=True)
A4 = pys.showdict(OD, rcsize=(rs, cs), stride=(0, 0), plot=False)

plt.figure()
plt.subplot(221)
plt.imshow(A1)
plt.colorbar()
plt.title('Incomplete: ' + str(ds1))
plt.subplot(222)
plt.imshow(A2)
plt.colorbar()
plt.title('Overcomplete: ' + str(ds2))
plt.subplot(223)
plt.imshow(A3)
plt.colorbar()
plt.title('Overcomplete: ' + str(ds3))
plt.subplot(224)
plt.imshow(A4)
plt.colorbar()
plt.title('Overcomplete: ' + str(ds4))
plt.tight_layout()
plt.show()

print("---------ODCT-2D-------------")
ds1 = (256, 64)
rs, cs = (int(np.sqrt(ds1[1])), int(np.sqrt(ds1[1])))
OD = pys.odctndict(dictshape=ds1, isnorm=True)
A1 = pys.showdict(OD, rcsize=(rs, cs), stride=(0, 0), plot=False)

ds2 = (256, 256)
rs, cs = (int(np.sqrt(ds2[1])), int(np.sqrt(ds2[1])))
OD = pys.odctndict(dictshape=ds2, isnorm=True)
A2 = pys.showdict(OD, rcsize=(rs, cs), stride=(0, 0), plot=False)

ds3 = (256, 1024)
rs, cs = (int(np.sqrt(ds3[1])), int(np.sqrt(ds3[1])))
OD = pys.odctndict(dictshape=ds3, axis=2, isnorm=True)
A3 = pys.showdict(OD, rcsize=(rs, cs), stride=(0, 0), plot=False)

ds4 = (256, 4096)
rs, cs = (int(np.sqrt(ds4[1])), int(np.sqrt(ds4[1])))
OD = pys.odctndict(dictshape=ds4, axis=2, isnorm=True)
A4 = pys.showdict(OD, rcsize=(rs, cs), stride=(0, 0), plot=False)

plt.figure()
plt.subplot(221)
plt.imshow(A1)
plt.colorbar()
plt.title('Incomplete: ' + str(ds1))
plt.subplot(222)
plt.imshow(A2)
plt.colorbar()
plt.title('Overcomplete: ' + str(ds2))
plt.subplot(223)
plt.imshow(A3)
plt.colorbar()
plt.title('Overcomplete: ' + str(ds3))
plt.subplot(224)
plt.imshow(A4)
plt.colorbar()
plt.title('Overcomplete: ' + str(ds4))
plt.tight_layout()
plt.show()
