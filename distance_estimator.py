import numpy as np
from numpy.linalg import norm as vecLen
from numba import jit

### numba bug workaround
### Can't create a numpy array from a numpy array
### https://github.com/numba/numba/issues/4470
from numba import types
from numba.extending import overload
@overload(np.array)
def np_array_ol(x):
	if isinstance(x, types.Array):
		def impl(x):
			return np.copy(x)
		return impl

from util import *

ITER = 3

@jit(nopython=True, nogil=True)
def floor(p):
	return p[2], np.array([240, 155, 223]) * 1.0

@jit(nopython=True, nogil=True)
def inf_ball(point):
	dist = point % 1.0 - 1.0 / 2
	return vecLen(dist) - 0.3, np.array([225, 181, 115]) * 1.0

@jit(nopython=True, nogil=True)
def tetrahedron(p):
	rotate = Rmat(np.array([-np.arctan(1 / np.sqrt(2)), 0, np.pi / 2]))
	rotate = rotate @ Rmat(np.array([0, -np.pi / 4, 0]))
	p = rotate.T @ p
	scale = 2.0
	ref_N = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]]) / np.sqrt(2)
	offset = np.array(U)
	for _ in range(ITER):
		p -= 2.0 * min(0.0, np.dot(p, ref_N[0])) * ref_N[0]
		p -= 2.0 * min(0.0, np.dot(p, ref_N[1])) * ref_N[1]
		p -= 2.0 * min(0.0, np.dot(p, ref_N[2])) * ref_N[2]
		p = (p - offset) * scale
	p /= scale
	tet_norm = np.array([[-1, 1, 1], [ 1,-1, 1], [ 1, 1,-1], [-1,-1,-1]]) / np.sqrt(3)
	tet_offs = np.array([[ 1, 1, 1], [ 1, 1, 1], [ 1, 1, 1], [ 1,-1,-1]])
	tet_dist = np.amax(np.sum((p - tet_offs) * tet_norm, axis=1))
	tet_dist *= pow(scale, float(-ITER + 1))
	return tet_dist, np.array([225, 181, 115]) * 1.0

@jit(nopython=True, nogil=True)
def tetrahedron_with_floor(p):
	offset = np.sqrt(3) * 2 / 3
	tet_dist, tet_color = tetrahedron(p - offset * UZ)
	floor_dist, floor_color = floor(p)
	if tet_dist < floor_dist:
		return tet_dist, tet_color
	else:
		return floor_dist, floor_color

@jit(nopython=True, nogil=True)
def cube(p):
	scale = 3.0
	vtx = np.array([
		[-1,-1,-1], [-1,-1, 0], [-1,-1, 1], [-1, 0,-1],
		[-1, 0, 1], [-1, 1,-1], [-1, 1, 0], [-1, 1, 1],
		[ 0,-1,-1], [ 0,-1, 1], [ 0, 1,-1], [ 0, 1, 1],
    [ 1,-1,-1], [ 1,-1, 0], [ 1,-1, 1], [ 1, 0,-1],
		[ 1, 0, 1], [ 1, 1,-1], [ 1, 1, 0], [ 1, 1, 1]
	])
	for _ in range(ITER):
		v = np.argmin(np.sum(np.square(vtx - p), axis=1))
		p = p * scale - vtx[v] * scale
	p /= scale
	cube_norm = np.array([[ 1, 0, 0], [-1, 0, 0], [ 0, 1, 0], [ 0,-1, 0], [ 0, 0, 1], [ 0, 0,-1]])
	cube_offs = np.array([[ 1, 1, 1], [-1,-1,-1], [ 1, 1, 1], [-1,-1,-1], [ 1, 1, 1], [-1,-1,-1]]) / 2
	cube_dist = np.amax(np.sum((p - cube_offs) * cube_norm, axis=1))
	cube_dist *= pow(scale, float(-ITER + 1))
	return cube_dist, np.array([225, 181, 115]) * 1.0

@jit(nopython=True, nogil=True)
def cube_with_floor(p):
	offset = 3/2
	cube_dist, cube_color = cube(p - offset * UZ)
	floor_dist, floor_color = floor(p)
	if cube_dist < floor_dist:
		return cube_dist, cube_color
	else:
		return floor_dist, floor_color

@jit(nopython=True, nogil=True)
def distorted_cube_with_floor(p):
	offset = 3/2
	distorted_p = p - offset * UZ
	distorted_p = Rmat((0, 0, distorted_p[2] * np.pi / 6)) @ distorted_p
	cube_dist, cube_color = cube(distorted_p)
	floor_dist, floor_color = floor(p)
	if cube_dist < floor_dist:
		return cube_dist, cube_color
	else:
		return floor_dist, floor_color

@jit(nopython=True, nogil=True)
def reflected_cube_with_floor(p):
	offset = 3/2
	plane_offset = np.array([ 0.4, 1.2, 0.9])
	plane_normal = np.array([-0.9,-1.3,-1.6])
	plane_normal = unit(plane_normal)
	reflected_p = p - offset * UZ
	reflected_p -= plane_offset
	reflected_p -= 2.0 * min(0.0, np.dot(reflected_p, plane_normal)) * plane_normal
	reflected_p += plane_offset
    
	cube_dist, cube_color = cube(reflected_p)
	floor_dist, floor_color = floor(p)
	if cube_dist < floor_dist:
		return cube_dist, cube_color
	else:
		return floor_dist, floor_color