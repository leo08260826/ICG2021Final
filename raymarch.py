from math import pow
from numpy import array, copy, dot, where
from numpy.linalg import norm
from numba import jit

### numba bug workaround
from numba import types
from numba.extending import overload

@overload(array)
def np_array_ol(x):
	if isinstance(x, types.Array):
		def impl(x):
			return copy(x)
		return impl

### parameter of ray marching
MaximumRaySteps = 128
MinimumDistance = 0.001

### light source
lightPos = array([10, 10, 10])
lightIntensity = 300

@jit(nopython=True, nogil=True)
def normalize(vec):
	return vec / norm(vec)

### Different Fractals have different DE and Normal functions!!!
@jit(nopython=True, nogil=True)
def DistanceEstimator(point):
	dist = array(point)
	dist[0] = point[0] % 1.0 - 0.5
	dist[1] = point[1] % 1.0 - 0.5
	return norm(dist) - 0.3

@jit(nopython=True, nogil=True)
def findNormal(point):
	N = array(point)
	N[0] = point[0] % 1.0 - 0.5
	N[1] = point[1] % 1.0 - 0.5
	return normalize(N)

### main function
@jit(nopython=True, nogil=True)
def rayMarching(pixelx, pixely, pixelz, directionx, directiony, directionz):
	pixel = array([pixelx, pixely, pixelz])
	direction = array([directionx, directiony, directionz])

	totalDistance = 0.0
	steps = 0
	distance = 0
	p = array(pixel)
	for steps in range(0, MaximumRaySteps):
		distance = DistanceEstimator(p)
		totalDistance += distance
		p = pixel + totalDistance * direction
		if(distance < MinimumDistance):
			break

	### render color
	if steps >= MaximumRaySteps: return array([0.0, 0.0, 0.0])
	steps_inter = steps + distance / MinimumDistance

	amb_color = array([255, 255, 255])
	amb_brightness = 0.1
	amb_occ = pow(2, -float(steps_inter)/float(MaximumRaySteps - 1) / amb_brightness)
	amb = amb_color * amb_occ

	N = findNormal(p)
	p2light = lightPos - p
	lightDistSq = dot(p2light, p2light)
	intensity = lightIntensity / lightDistSq

	dif_ratio = max(dot(N, normalize(p2light)), 0.0)
	dif = amb * dif_ratio * intensity

	spc_dir = normalize(p2light) - normalize(p - array([0, 0, 4]))
	spc_ratio = max(dot(N, normalize(spc_dir)), 0.0)
	spc = array([255, 255, 255]) * pow(spc_ratio, 8) * intensity

	renderColor = dif * 0.05 + dif + spc * .2
	renderColor = where(renderColor > 255, 255, renderColor)
	renderColor = where(renderColor < 0, 0, renderColor)

	return array(renderColor)
