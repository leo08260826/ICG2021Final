from math import pow, sqrt

import cupy as np
from cupy.linalg import norm as vecLen

### params and uitls
from params import *
from util import *

### Fractals DE
def DE_balls(point, space, radius):
	dist = point % space - space / 2
	# dist[2] = point[2]
	return vecLen(dist, axis = -1) - radius

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
def normal_inf_ball(point, space):
	dist = point % space - space / 2
	# dist[2] = point[2]
	return unit(dist)

### unused
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
def rayMarching(pixelData):
	[pixel, dir] = pixelData

	### choose DE
	DE = lambda point: DE_balls(point, 1.0, 0.3)

	### march
	totalDistance = 0.0
	distance = 0
	point = np.array(pixel)
	step_arr = np.full(pixel.shape[0], 0)
	activate = np.full(pixel.shape[0], True)
	for _ in range(0, MAX_STEP):
		distance = np.where(activate, DE(point), distance)
		totalDistance += np.where(activate, distance, 0)
		point = np.where(activate[:, np.newaxis], pixel + totalDistance[:, np.newaxis] * dir, point)
		step_arr += np.where(activate, 1, 0)
		activate = np.where(np.logical_and(activate, (distance >= MIN_DIST)), True, False)
		if(not activate.any()):
			break

	### interpolate steps to smooth the color rendering
	# if steps >= (MAX_STEP - 1): return O
	steps_inter = step_arr + distance / MIN_DIST * 1.0

	### params
	N = normal_inf_ball(point, 1.0)
	# check if normal faces camera, if not then reverse
	# if np.dot(N, point - CAM_POS) > 0: N = -N
	p2light = L_POS - point
	lightDistSq = np.einsum('ij,ij->i', p2light, p2light)
	intensity = L_ITEN / lightDistSq

	### Ambient
	amb_color = WHITE
	amb_occ = np.power(2, -steps_inter / float(MAX_STEP - 1) / AMB_I)
	amb = amb_color * amb_occ[:, np.newaxis]

	### Diffusion
	dif_ratio = np.maximum(np.einsum('ij,ij->i', N, unit(p2light)), 0.0)
	dif = amb_color * dif_ratio[:, np.newaxis] * intensity[:, np.newaxis]

	### Specular
	spc_dir = unit(p2light) - unit(point - CAM_POS)
	spc_ratio = np.maximum(np.einsum('ij,ij->i', N, unit(spc_dir)), 0.0)
	spc = WHITE * np.power(spc_ratio, SPC_P)[:, np.newaxis] * intensity[:, np.newaxis]

	### combining
	renderColor = amb * AMB_R + dif * DIF_R + spc * SPC_R
	renderColor = np.where(renderColor > 255, 255, renderColor)
	renderColor = np.where(renderColor < 0, 0, renderColor)
	return renderColor
