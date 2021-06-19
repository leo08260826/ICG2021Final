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

### constants
ux = np.array([1.0, 0.0, 0.0])
uy = np.array([0.0, 1.0, 0.0])
uz = np.array([0.0, 0.0, 1.0])
O = np.array([0.0, 0.0, 0.0])
I = np.array([1.0, 1.0, 1.0])
WHITE = np.array([255.0, 255.0, 255.0])

### parameters
MAX_STEP = 16
MIN_DIST = 0.001

CAM_POS = np.array([0.0, 0.0, 4.0])

L_POS = np.array([10.0, 10.0, 10.0])
L_ITEN = 300.0

AMB_I = 0.1
AMB_R = 0.05
DIF_R = 1.0
SPC_R = 0.2
SPC_P = 8.0

### utilities
@jit(nopython=True, nogil=True)
def unit(vec): return vec / vecLen(vec)

### Fractals DE
@jit(nopython=True, nogil=True)
def DE_balls(point, space, radius):
	dist = point % space - space / 2
	dist[2] = point[2]
	return vecLen(dist) - radius

@jit(nopython=True, nogil=True)
def DE_tetrahedron(point):
	iter = 5
	scale = 2.0
	reflect_n = np.array([[1,1,0],[1,0,1],[0,1,1]]) / sqrt(2)
	offset = np.array([1,1,1])
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
# 	dx = ux * d
# 	dy = uy * d
# 	dz = uz * d
# 	return unit(np.array([
# 		DE(point + dx) - DE(point - dx),
# 		DE(point + dy) - DE(point - dy),
# 		DE(point + dz) - DE(point - dz)
# 	]))

# fast normal finder for balls
@jit(nopython=True, nogil=True)
def normal_inf_ball(point):
	N = np.array(point)
	N[0] = point[0] % 1.0 - 0.5
	N[1] = point[1] % 1.0 - 0.5
	return unit(N)

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

	### render color
	if steps >= (MAX_STEP - 1): return O
	steps_inter = steps + distance / MIN_DIST

	N = normal_inf_ball(point)
	if np.dot(N, point - CAM_POS) > 0: N = -N
	p2light = L_POS - point
	lightDistSq = np.dot(p2light, p2light)
	intensity = L_ITEN / lightDistSq

	amb_color = WHITE
	amb_occ = pow(2, -float(steps_inter)/float(MAX_STEP - 1) / AMB_I)
	amb = amb_color * amb_occ

	dif_ratio = max(np.dot(N, unit(p2light)), 0.0)
	dif = amb_color * dif_ratio * intensity

	spc_dir = unit(p2light) - unit(point - CAM_POS)
	spc_ratio = max(np.dot(N, unit(spc_dir)), 0.0)
	spc = WHITE * pow(spc_ratio, SPC_P) * intensity

	renderColor = amb * AMB_R + dif * DIF_R + spc * SPC_R
	renderColor = np.where(renderColor > 255, 255, renderColor)
	renderColor = np.where(renderColor < 0, 0, renderColor)
	return renderColor
