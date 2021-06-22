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

WHITE = U * 255
BLACK = O * 255

### utilities
@jit(nopython=True, nogil=True)
def unit(vec):
	return vec / norm(vec)

### using Tait–Bryan angles
@jit(nopython=True, nogil=True)
def Rmat(rpy):
	roll, pitch, yaw = rpy
	roll, pitch, yaw = -roll, pitch, -yaw
	Rx = np.array([[1.0, 0.0, 0.0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
	Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0.0, 1.0, 0.0], [-np.sin(pitch), 0, np.cos(pitch)]])
	Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0.0, 0.0, 1.0]])
	return Rz @ Ry @ Rx

@jit(nopython=True, nogil=True)
def reflect(N, V):
	return 2 * np.dot(N, V) * N - V
