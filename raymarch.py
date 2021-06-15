from numpy import array
from numpy.linalg import norm
from numba import jit
import numba

### parameter of ray marching
MaximumRaySteps = 10
MinimumDistance = 0.01

@jit(nopython=True, nogil=True)
def DistanceEstimator(point):
	point[0] = point[0]%1.0 - 0.5
	point[1] = point[1]%1.0 - 0.5
	return norm(point)-0.3

@jit(nopython=True, nogil=True)
def rayMarching(pixelx, pixely, pixelz, directionx, directiony, directionz):
	maxiumRaySteps = MaximumRaySteps
	minimumDistance = MinimumDistance
	pixel = array([pixelx, pixely, pixelz])
	direction = array([directionx, directiony, directionz])

	totalDistance = 0.0
	steps = 0
	for steps in range(0, maxiumRaySteps):
		p = pixel + totalDistance * direction;
		distance = DistanceEstimator(p);
		totalDistance += distance;
		if(distance < minimumDistance):
			break

	### use number of steps as render color (gray scale)
	tmp = (1.0-float(steps)/float(maxiumRaySteps))*255
	return array([tmp, tmp, tmp])