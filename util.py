import numpy as np
from numpy.linalg import norm
from numba import jit

### constants
O = np.array([0.0, 0.0, 0.0])
UX = np.array([1.0, 0.0, 0.0])
UY = np.array([0.0, 1.0, 0.0])
UZ = np.array([0.0, 0.0, 1.0])
U = np.array([1.0, 1.0, 1.0])
I = np.array([UX, UY, UZ])

WHITE = U
BLACK = O

### utilities
@jit(nopython=True, nogil=True)
def unit(vec):
	return vec / norm(vec)

@jit(nopython=True, nogil=True)
def Rmat(x, y, z):
	Rx = np.array([[1.0, 0.0, 0.0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
	Ry = np.array([[np.cos(y), 0, np.sin(y)], [0.0, 1.0, 0.0], [-np.sin(y), 0, np.cos(y)]])
	Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0.0, 0.0, 1.0]])
	return Rx @ Ry @ Rz

@jit(nopython=True, nogil=True)
def reflect(N, V):
	return 2 * np.dot(N, V) * N - V
