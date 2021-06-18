import numpy as np
from numba import jit

### parameter of ray marching
MaximumRaySteps = 10
MinimumDistance = 0.01

@jit(nopython=True, nogil=True)
def DE_inf_ball(point):
	point[0] = point[0]%1.0 - 0.5
	point[1] = point[1]%1.0 - 0.5
	return np.linalg.norm(point)-0.3

@jit(nopython=True, nogil=True)
def DE_tetrahedron(point):
	iteration = 10
	scale = 2.0
	reflect_normals = np.array([[1,1,0],[1,0,1],[0,1,1]]) / np.sqrt(2)
	offset = np.array([1,1,1])
	for _ in range(iteration):
		point -= 2.0 * min(0.0, np.dot(point, reflect_normals[0])) * reflect_normals[0]
		point -= 2.0 * min(0.0, np.dot(point, reflect_normals[1])) * reflect_normals[1]
		point -= 2.0 * min(0.0, np.dot(point, reflect_normals[2])) * reflect_normals[2]
		point = point * scale - offset * (scale - 1.0)
	return np.linalg.norm(point) * pow(scale, float(-iteration))

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

@jit(nopython=True, nogil=True)
def rayMarching(pixel, direction, object_type='tetrahedron'):
	if object_type == 'tetrahedron':
		DistanceEstimator = DE_tetrahedron
	else:
		DistanceEstimator = DE_inf_ball
  
	totalDistance = 0.0
	steps = 0
	for steps in range(0, MaximumRaySteps):
		p = pixel + totalDistance * direction;
		distance = DistanceEstimator(p);
		totalDistance += distance;
		if(distance < MinimumDistance):
			break

	### use number of steps as render color (gray scale)
	tmp = (1.0-float(steps)/float(MaximumRaySteps))*255
	return np.array([tmp, tmp, tmp])