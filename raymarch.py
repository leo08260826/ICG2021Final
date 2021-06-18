from math import pow
from numpy import array
from numpy.linalg import norm
from numba import jit

### parameter of ray marching
MaximumRaySteps = 128
MinimumDistance = 0.001

@jit(nopython=True, nogil=True)
def DistanceEstimator(point):
	point[0] = point[0]%1.0 - 0.5
	point[1] = point[1]%1.0 - 0.5
	return norm(point)-0.3

@jit(nopython=True, nogil=True)
def rayMarching(pixelx, pixely, pixelz, directionx, directiony, directionz):
	pixel = array([pixelx, pixely, pixelz])
	direction = array([directionx, directiony, directionz])

	totalDistance = 0.0
	steps = 0
	for steps in range(0, MaximumRaySteps):
		p = pixel + totalDistance * direction;
		distance = DistanceEstimator(p);
		totalDistance += distance;
		if(distance < MinimumDistance):
			break

	### use number of steps as render color (gray scale)
	amb_brightness = 0.1
	amb_ratio = pow(2, -float(steps)/float(MaximumRaySteps - 1) / amb_brightness)
	if steps == MaximumRaySteps: amb_ratio = 0.0
	amb = amb_ratio * 255
	return array([amb, amb, amb])