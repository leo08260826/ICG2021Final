import numpy as np
from numpy.linalg import norm
from numba import jit

### constants
UX = np.array([1.0, 0.0, 0.0])
UY = np.array([0.0, 1.0, 0.0])
UZ = np.array([0.0, 0.0, 1.0])
O = np.array([0.0, 0.0, 0.0])
I = np.array([1.0, 1.0, 1.0])
WHITE = np.array([255.0, 255.0, 255.0])

### utilities
@jit(nopython=True, nogil=True)
def unit(vec):
	return vec / norm(vec)

@jit(nopython=True, nogil=True)
def rotation_matrix(x, y, z):
	Rx = np.array([[1.0,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
	Ry = np.array([[np.cos(y),0,np.sin(y)],[0,1.0,0],[-np.sin(y),0,np.cos(y)]])
	Rz = np.array([[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1.0]])
	return Rx @ Ry @ Rz

@jit(nopython=True, nogil=True)
def hsv_to_rgb(h, s, v):
	if s == 0.0: return np.array([v, v, v])
	i = int(h*6.) # XXX assume int() truncates!
	f = (h*6.)-i; p,q,t = v*(1.-s), v*(1.-s*f), v*(1.-s*(1.-f)); i%=6
	if i == 0: return np.array([v, t, p])
	if i == 1: return np.array([q, v, p])
	if i == 2: return np.array([p, v, t])
	if i == 3: return np.array([p, q, v])
	if i == 4: return np.array([t, p, v])
	if i == 5: return np.array([v, p, q])