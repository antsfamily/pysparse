import numpy as np


a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

b = np.reshape(a, (2, 2, 1, 3))


print(a)

print(b[:, :, 0, 0])
print(b[:, :, 0, 1])
print(b[:, :, 0, 2])


a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])


i, j = np.mgrid[0:2, 2:4]

print(a)
print(a[i, j])

print(a)
print(np.kron(a, a).shape)
print(np.kron(np.kron(a, a), a).shape)


print("---split")

a = a.transpose()
b = np.split(a, 2)
print(a)
print(b[0])
print(b[1])
