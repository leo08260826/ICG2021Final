import numpy as np
from util import *
from numba import jit

fractal_iteration = 3

@jit(nopython=True, nogil=True)
def inf_ball(point):
	point[0] = point[0]%1.0 - 0.5
	point[1] = point[1]%1.0 - 0.5
	return np.linalg.norm(point)-0.3, np.array([225/255, 181/255, 115/255])

# @jit(nopython=True, nogil=True)
# def DE_balls(point, space, radius):
# 	dist = point % space - space / 2
# 	# dist[2] = point[2]
# 	return vecLen(dist) - radius

@jit(nopython=True, nogil=True)
def tetrahedron(p):
	p = rotation_matrix(np.arctan(1/np.sqrt(2)),-np.pi/4,0).T @ p
	scale = 2.0
	reflect_normals = np.array([[1,1,0],[1,0,1],[0,1,1]]) / np.sqrt(2)
	offset = np.array([1,1,1])
	for _ in range(fractal_iteration):
		p -= 2.0 * min(0.0, np.dot(p, reflect_normals[0])) * reflect_normals[0]
		p -= 2.0 * min(0.0, np.dot(p, reflect_normals[1])) * reflect_normals[1]
		p -= 2.0 * min(0.0, np.dot(p, reflect_normals[2])) * reflect_normals[2]
		p = p * scale - offset * scale
	p /= scale
	tet_norm = np.array([[-1,1,1],[1,-1,1],[1,1,-1],[-1,-1,-1]]) / np.sqrt(3)
	tet_offs = np.array([[1,1,1],[1,1,1],[1,1,1],[1,-1,-1]])
	tet_dist = ((p - tet_offs) * tet_norm).sum(axis=1).max()
	return tet_dist * pow(scale, float(-fractal_iteration+1)), np.array([225/255, 181/255, 115/255])

# @jit(nopython=True, nogil=True)
# def DE_tetrahedron(point):
# 	iter = 5
# 	scale = 2.0
# 	reflect_n = np.array([UX + UY, UY + UZ, UX + UZ]) / sqrt(2)
# 	offset = U
# 	for _ in range(iter):
# 		point -= 2.0 * min(0.0, np.dot(point, reflect_n[0])) * reflect_n[0]
# 		point -= 2.0 * min(0.0, np.dot(point, reflect_n[1])) * reflect_n[1]
# 		point -= 2.0 * min(0.0, np.dot(point, reflect_n[2])) * reflect_n[2]
# 		point = (point - offset) * scale
# 	return vecLen(point) * pow(scale, float(-iter))

@jit(nopython=True, nogil=True)
def cube(p):
	scale = 3.0
	vtx = np.array([[-1,-1,-1],[-1,-1,0],[-1,-1,1],[-1,0,-1],[-1,0,1],[-1,1,-1],[-1,1,0],[-1,1,1],\
                    [0,-1,-1],[0,-1,1],[0,1,-1],[0,1,1],\
                    [1,-1,-1],[1,-1,0],[1,-1,1],[1,0,-1],[1,0,1],[1,1,-1],[1,1,0],[1,1,1]])
	for _ in range(fractal_iteration):
		v = np.square(vtx - p).sum(axis=1).argmin()
		p = p * scale - vtx[v] * scale
	p /= scale
	cube_norm = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
	cube_offs = np.array([[1,1,1],[-1,-1,-1],[1,1,1],[-1,-1,-1],[1,1,1],[-1,-1,-1]]) / 2
	cube_dist = ((p - cube_offs) * cube_norm).sum(axis=1).max()
	return cube_dist * pow(scale, float(-fractal_iteration+1)), np.array([225/255, 181/255, 115/255])

@jit(nopython=True, nogil=True)
def floor(p):
	return p[2], np.array([240/255, 155/255, 223/255])

@jit(nopython=True, nogil=True)
def tetrahedron_with_floor(p):
	offset = np.sqrt(3) * 2 / 3
	tet_dist, tet_color = tetrahedron(p - np.array([0,0,offset]))
	floor_dist, floor_color = floor(p)
	if tet_dist < floor_dist:
		return tet_dist, tet_color
	else:
		return floor_dist, floor_color

@jit(nopython=True, nogil=True)
def cube_with_floor(p):
	offset = 3/2
	cube_dist, cube_color = cube(p - np.array([0,0,offset]))
	floor_dist, floor_color = floor(p)
	if cube_dist < floor_dist:
		return cube_dist, cube_color
	else:
		return floor_dist, floor_color

@jit(nopython=True, nogil=True)
def distorted_cube_with_floor(p):
	offset = 3/2
	distorted_p = p - np.array([0,0,offset])
	distorted_p = rotation_matrix(0, 0, -distorted_p[2]*np.pi/6) @ distorted_p
	cube_dist, cube_color = cube(distorted_p)
	floor_dist, floor_color = floor(p)
	if cube_dist < floor_dist:
		return cube_dist, cube_color
	else:
		return floor_dist, floor_color

@jit(nopython=True, nogil=True)
def reflected_cube_with_floor(p):
	offset = 3/2
	plane_offset = np.array([0.4,1.2,0.9])
	plane_normal = np.array([-0.9,-1.3,-1.6])
	plane_normal /= np.linalg.norm(plane_normal)
	reflected_p = p - np.array([0,0,offset])
	reflected_p -= plane_offset
	reflected_p -= 2.0 * min(0.0, np.dot(reflected_p, plane_normal)) * plane_normal
	reflected_p += plane_offset
    
	cube_dist, cube_color = cube(reflected_p)
	floor_dist, floor_color = floor(p)
	if cube_dist < floor_dist:
		return cube_dist, cube_color
	else:
		return floor_dist, floor_color