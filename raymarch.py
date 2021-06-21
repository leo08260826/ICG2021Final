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

import distance_estimator as de

LightPosition = np.array([10.0, 10.0, 10.0])
LightColor = np.array([1.0, 1.0, 1.0])
specular_exp = 8
reflect_decay = 0.2
shadow_softness = 16

w_occlusion = 0.7
w_softshadow = 0.4

w_ambient = 0.1
w_diffuse = 0.5
w_specular = 0.4

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
def ambient(obj_color):
	return obj_color

@jit(nopython=True, nogil=True)
def occlusion(obj_color, steps, distance):
	steps_inter = steps + distance / MIN_DIST
	occ_factor = pow(2, -float(steps_inter)/float(MAX_STEP - 1) / 0.1)
	return occ_factor

@jit(nopython=True, nogil=True)
def diffuse(obj_color, N, p):
	L = LightPosition - p
	intensity = LightColor.sum()/ 3 / np.square(np.linalg.norm(L))
	L /= np.linalg.norm(L)
	return LightColor * obj_color * max(np.dot(N, L), 0.0) #* intensity

@jit(nopython=True, nogil=True)
def specular(N, p, V):
	L = LightPosition - p
	intensity = LightColor.sum()/ 3 #/ np.square(np.linalg.norm(L))
	L /= np.linalg.norm(L)
	R = reflect_direction(N, L)
	return LightColor * pow(max(np.dot(R, V), 0.0), specular_exp) #* intensity

@jit(nopython=True, nogil=True)
def softshadow(init, DE):
	direction = LightPosition - init
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
@jit(nopython=True, nogil=True)
def rayMarching(pixel, dir, DE=de.inf_ball, reflect_count=10):

	### choose DE
	# DE = lambda point: DE_balls(point, 1.0, 0.3)

	### march
	totalDistance = 0.0
	steps = 0
	distance = 0
	point = np.array(pixel)
	for steps in range(0, MAX_STEP):
		# point = pixel + totalDistance * dir
		distance, c_obj = DE(point)
		totalDistance += distance
		point = pixel + totalDistance * dir
		if(distance < MIN_DIST):
			break

	# c_ambient = ambient(c_obj)       
	# c_diffuse = diffuse(c_obj, N, p)
	# c_specular = specular(N, p, -direction)
			
	# f_occlusion = occlusion(c_obj, steps, distance) * w_occlusion + (1-w_occlusion)

	# color = np.array([0.0, 0.0, 0.0])
	# color += w_ambient * c_ambient * f_occlusion
	# color += w_diffuse * c_diffuse * f_softshadow
	# color += w_specular * c_specular * f_softshadow

	### interpolate steps to smooth the color rendering
	if steps >= (MAX_STEP - 1): return background_color(point)
	steps_inter = steps + distance / MIN_DIST

	### params
	N = normal_inf_ball(point, 1.0)
	N = unit(N)
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

	SOFT_SH = softshadow(point, DE) * w_softshadow + (1-w_softshadow)

	### combining
	renderColor = amb * AMB_R + dif * DIF_R * SOFT_SH + spc * SPC_R * SOFT_SH

	# if reflect_count > 0 and np.dot(N, -dir) > 0:
	# 	reflect_direct = reflect_direction(N, -dir)
	# 	reflect_p = point + reflect_direct * MIN_DIST
	# 	reflect_color = rayMarching(reflect_p, reflect_direct, DE, reflect_count-1)
	# 	renderColor = (1-reflect_decay) * renderColor + reflect_decay * reflect_color

	return renderColor
