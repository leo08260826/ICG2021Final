from math import pow

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

import distance_estimator as de

reflect_decay = 0.5
shadow_softness = 16
w_softshadow = 0.4

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
def normal_inf_ball(point):
	dist = point % 1.0 - 1.0 / 2
	# dist[2] = point[2]
	return unit(dist)

@jit(nopython=True, nogil=True)
def normal(p, DE):
	dp = np.array([[MIN_DIST/10,0,0],[0,MIN_DIST/10,0],[0,0,MIN_DIST/10]])
	px, _ = DE(p+dp[0])
	nx, _ = DE(p-dp[0])
	py, _ = DE(p+dp[1])
	ny, _ = DE(p-dp[1])
	pz, _ = DE(p+dp[2])
	nz, _ = DE(p-dp[2])
	return np.array([px-nx, py-ny, pz-nz])

@jit(nopython=True, nogil=True)
def reflect_direction(N, X):
	return 2 * np.dot(N, X) * N - X

@jit(nopython=True, nogil=True)
def background_color(d):
	d /= np.linalg.norm(d)
	w = - d[0] + d[1] - d[2]
	w += 3 / np.sqrt(3)
	w /= 2 * 3 / np.sqrt(3)
	return np.array([172/255, 253/255, 240/255]) * w + np.array([62/255, 67/255, 218/255]) * (1-w)

@jit(nopython=True, nogil=True)
def occ(steps):
	ratio = pow(2, -float(steps)/float(MAX_STEP - 1) / AMB_I)
	return ratio

@jit(nopython=True, nogil=True)
def dif(c, N, point):
	p2light = L_POS - point
	dif_ratio = max(np.dot(N, unit(p2light)), 0.0)
	return c * dif_ratio

@jit(nopython=True, nogil=True)
def spc(c, N, point):
	spc_dir = unit(L_POS - point) - unit(point - CAM_POS)
	spc_ratio = max(np.dot(N, unit(spc_dir)), 0.0)
	spc_ratio = c * pow(spc_ratio, SPC_P)
	return c * spc_ratio

@jit(nopython=True, nogil=True)
def softshadow(init, DE):
	direction = L_POS - init
	direction /= np.linalg.norm(direction)
	totalDistance = MIN_DIST # avoid division by zero
	init += totalDistance * direction
	steps = 0
	illum = 1.0
	for steps in range(0, MAX_STEP):
		p = init + totalDistance * direction
		distance, c_obj = DE(p)
		illum = max(0.0, min(illum, shadow_softness * distance / totalDistance))
		totalDistance += distance
		if(distance < MIN_DIST):
			break
	return illum

### main function
def rayMarching(pixel, dir, debug, DE=de.tetrahedron_with_floor, reflect_count=7):

	### choose DE
	# DE = lambda point: DE_balls(point, 1.0, 0.3)

	### march
	totalDistance = 0.0
	steps = 0
	distance = 0
	point = np.array(pixel)
	color = BLACK
	for steps in range(0, MAX_STEP):
		# point = pixel + totalDistance * dir
		distance, color = DE(point)
		totalDistance += distance
		point = pixel + totalDistance * dir
		if(distance < MIN_DIST):
			break

	### interpolate steps to smooth the color rendering
	if steps >= (MAX_STEP - 1): return background_color(point)
	steps_inter = steps + distance / MIN_DIST

	### params
	N = normal(point, DE)
	N = unit(N)
	# check if normal faces camera, if not then reverse
	# if np.dot(N, point - CAM_POS) > 0: N = -N
	p2light = L_POS - point
	lightDistSq = np.dot(p2light, p2light)
	intensity = L_ITEN / lightDistSq
	# softShadow_r = softshadow(point, DE) * w_softshadow + (1-w_softshadow)
	softShadow_r = 1.0

	renderColor = color * AMB_R * occ(steps_inter) * 1
	renderColor += dif(color, N, point) * DIF_R * intensity * softShadow_r
	renderColor += spc(color, N, point) * SPC_R * intensity * softShadow_r

	if reflect_count > 0 and np.dot(N, -dir) > 0:
		reflect_direct = reflect_direction(N, -dir)
		reflect_p = point + reflect_direct * MIN_DIST * 2
		reflect_color = rayMarching(reflect_p, reflect_direct, False, DE, reflect_count-1)
		renderColor = (1 - reflect_decay) * renderColor + reflect_decay * reflect_color

	# return (N + 1) / 2.0
	return renderColor
	# return U * steps_inter / 256.0
