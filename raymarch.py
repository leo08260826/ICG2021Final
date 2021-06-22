from math import pow

import numpy as np
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

from params import *
from util import *

### general normal function
@jit(nopython=True, nogil=True)
def getNormal(point, DE):
	d = MIN_DIST / 10
	dx = UX * d
	dy = UY * d
	dz = UZ * d
	return unit(np.array([
		DE(point + dx)[0] - DE(point - dx)[0],
		DE(point + dy)[0] - DE(point - dy)[0],
		DE(point + dz)[0] - DE(point - dz)[0]
	]))

### fast normal finder for balls
@jit(nopython=True, nogil=True)
def normal_inf_ball(point):
	dist = point % 1.0 - 1.0 / 2
	return unit(dist)

### color render
@jit(nopython=True, nogil=True)
def background_color(point):
	w = np.sum(unit(point) * np.array([-1, 1, -1]))
	w = w / (2 * np.sqrt(3)) + 0.5
	sky = np.array([172, 253, 240])
	floor = np.array([62, 67, 218])
	return sky * w + floor * (1 - w)

@jit(nopython=True, nogil=True)
def occ(steps):
	ratio = pow(2, -float(steps)/float(MAX_STEP - 1) / AMB_I)
	return ratio

@jit(nopython=True, nogil=True)
def dif(point, N, color):
	dif_ratio = max(np.dot(N, unit(L_POS - point)), 0.0)
	return color * dif_ratio

@jit(nopython=True, nogil=True)
def spc(point, N, color):
	spc_dir = unit(L_POS - point) - unit(point - CAM_POS)
	spc_ratio = max(np.dot(N, unit(spc_dir)), 0.0)
	return color * pow(spc_ratio, SPC_P)

@jit(nopython=True, nogil=True)
def renderColor(point, N, color, DE, steps):
	intensity = L_ITEN / np.dot(L_POS - point, L_POS - point)
	_, _, _, softShadow = march(point, unit(L_POS - point), DE, shadow=True)
	softShadow_r = softShadow * SSH_I + (1 - SSH_I)

	amb_color = color * occ(steps)
	rendered = amb_color * AMB_R
	rendered += dif(point, N, amb_color) * DIF_R * intensity * softShadow_r
	rendered += spc(point, N, amb_color) * SPC_R * intensity * softShadow_r
	return rendered

### core of ray marching
@jit(nopython=True, nogil=True)
def march(pixel, direction, DE, shadow=False):
	totalDistance = 0.0
	steps = 0
	distance = 0
	point = np.array(pixel)
	color = BLACK
	ill = 1.0
	for steps in range(0, MAX_STEP):
		# point = pixel + totalDistance * direction
		distance, color = DE(point)
		totalDistance += distance
		point = pixel + totalDistance * direction
		if shadow:
			ill = max(0.0, min(ill, SSH_S * distance / totalDistance))
		if(distance < MIN_DIST): break
	### interpolate steps to smooth the color rendering
	steps_inter = steps + distance / MIN_DIST
	return steps_inter, point, color, ill

### numba does not support recursive function call
def getReflectColor(point, N, color, direction, DE, count):
	reflect_dir = reflect(N, -direction)
	reflect_point = point + reflect_dir * MIN_DIST * 2
	reflect_color = rayMarching(reflect_point, reflect_dir, DE=DE, reflect_count=count-1)
	color = (1 - REF_R) * color + REF_R * reflect_color
	return color

### main function
def rayMarching(pixel, direction, DE=FRACTAL, reflect_count=7):
	steps, point, color, _ = march(pixel, direction, DE)

	if steps >= (MAX_STEP - 1): return background_color(point)

	N = getNormal(point, DE)

	color = renderColor(point, N, color, DE, steps)

	if reflect_count > 0 and np.dot(N, -direction) > 0:
		color = getReflectColor(point, N, color, direction, DE, reflect_count)

	return color
