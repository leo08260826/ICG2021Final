import numpy as np
from numpy.linalg import norm as vecLen

from numba import jit

from util import *

fractal_iteration = 4

@jit(nopython=True, nogil=True)
def inf_ball(point):
	dist = point % 1.0 - 1.0 / 2
	return vecLen(dist) - 0.3, np.array([225, 181, 115])/255.0

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
def halved_cube_with_floor(p):
	offset = np.array([0,0,3/2])
	knife_offset = np.array([0,0,2.0])
	knife_normal = np.array([-1.0,-1.0,3.0])
	knife_normal /= np.linalg.norm(knife_normal)
	cube_dist, cube_color = cube(p - offset)
	floor_dist, floor_color = floor(p)
	knife_dist = np.dot(knife_normal, p - knife_offset)
	if knife_dist > cube_dist:
		cube_dist = knife_dist
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
