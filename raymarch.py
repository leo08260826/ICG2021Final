from math import pow, sqrt

import numpy as np
from numpy.linalg import norm as vecLen
from numba import jit

### numba bug workaround
from numba import types
from numba.extending import overload

@overload(np.array)
def np_array_ol(x):
	if isinstance(x, types.Array):
		def impl(x):
			return np.copy(x)
		return impl

### params and uitls
from params import *
from util import *

### Fractals DE
@jit(nopython=True, nogil=True)
def DE_balls(point, space, radius):
	dist = point % space - space / 2
	# dist[2] = point[2]
	return vecLen(dist) - radius

@jit(nopython=True, nogil=True)
def DE_tetrahedron(point):
	iter = 5
	scale = 2.0
	reflect_n = np.array([UX + UY, UY + UZ, UX + UZ]) / sqrt(2)
	offset = I
	for _ in range(iter):
		point -= 2.0 * min(0.0, np.dot(point, reflect_n[0])) * reflect_n[0]
		point -= 2.0 * min(0.0, np.dot(point, reflect_n[1])) * reflect_n[1]
		point -= 2.0 * min(0.0, np.dot(point, reflect_n[2])) * reflect_n[2]
		point = (point - offset) * scale
	return vecLen(point) * pow(scale, float(-iter))

### general normal function
# @jit(nopython=True, nogil=True)
# def findNormal(point, DE):
# 	d = MIN_DIST / 2
# 	dx = UX * d
# 	dy = UY * d
# 	dz = UZ * d
# 	return unit(np.array([
# 		DE(point + dx) - DE(point - dx),
# 		DE(point + dy) - DE(point - dy),
# 		DE(point + dz) - DE(point - dz)
# 	]))

### fast normal finder for balls
@jit(nopython=True, nogil=True)
def normal_inf_ball(point, space):
	dist = point % space - space / 2
	# dist[2] = point[2]
	return unit(dist)

### unused
# @jit(nopython=True, nogil=True)
# def hsv_to_rgb(h, s, v):
# 	if s == 0.0: return np.array([v, v, v])
# 	i = int(h*6.) # XXX assume int() truncates!
# 	f = (h*6.)-i; p,q,t = v*(1.-s), v*(1.-s*f), v*(1.-s*(1.-f)); i%=6
# 	if i == 0: return np.array([v, t, p])
# 	if i == 1: return np.array([q, v, p])
# 	if i == 2: return np.array([p, v, t])
# 	if i == 3: return np.array([p, q, v])
# 	if i == 4: return np.array([t, p, v])
# 	if i == 5: return np.array([v, p, q])

### main function
@jit(nopython=True, nogil=True)
def rayMarching(pixel, dir):

	### choose DE
	DE = lambda point: DE_balls(point, 1.0, 0.3)

	### march
	totalDistance = 0.0
	steps = 0
	distance = 0
	point = np.array(pixel)
	for steps in range(0, MAX_STEP):
		distance = DE(point)
		totalDistance += distance
		point = pixel + totalDistance * dir
		if(distance < MIN_DIST):
			break

	### interpolate steps to smooth the color rendering
	if steps >= (MAX_STEP - 1): return O
	steps_inter = steps + distance / MIN_DIST

	### params
	N = normal_inf_ball(point, 1.0)
	# check if normal faces camera, if not then reverse
	# if np.dot(N, point - CAM_POS) > 0: N = -N
	p2light = L_POS - point
	lightDistSq = np.dot(p2light, p2light)
	intensity = L_ITEN / lightDistSq

	### Ambient
	amb_color = WHITE
	amb_occ = pow(2, -float(steps_inter)/float(MAX_STEP - 1) / AMB_I)
	amb = amb_color * amb_occ

	### Diffusion
	dif_ratio = max(np.dot(N, unit(p2light)), 0.0)
	dif = amb_color * dif_ratio * intensity

	### Specular
	spc_dir = unit(p2light) - unit(point - CAM_POS)
	spc_ratio = max(np.dot(N, unit(spc_dir)), 0.0)
	spc = WHITE * pow(spc_ratio, SPC_P) * intensity

	### combining
	renderColor = amb * AMB_R + dif * DIF_R + spc * SPC_R
	renderColor = np.where(renderColor > 255, 255, renderColor)
	renderColor = np.where(renderColor < 0, 0, renderColor)
	return renderColor
