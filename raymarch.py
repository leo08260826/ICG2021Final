import numpy as np
from numba import jit

### parameter of ray marching
MaximumRaySteps = 20
MinimumDistance = 0.001

@jit(nopython=True, nogil=True)
def rotation_matrix(x, y, z):
	Rx = np.array([[1.0,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
	Ry = np.array([[np.cos(y),0,np.sin(y)],[0,1.0,0],[-np.sin(y),0,np.cos(y)]])
	Rz = np.array([[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1.0]])
	return Rx @ Ry @ Rz

@jit(nopython=True, nogil=True)
def DE_inf_ball(point):
	point[0] = point[0]%1.0 - 0.5
	point[1] = point[1]%1.0 - 0.5
	return np.linalg.norm(point)-0.3

@jit(nopython=True, nogil=True)
def DE_tetrahedron(p):
	p[2] += 6
	p = rotation_matrix(0.48, 0.7, 0.29) @ p
	iteration = 10
	scale = 2.0
	reflect_normals = np.array([[1,1,0],[1,0,1],[0,1,1]]) / np.sqrt(2)
	offset = np.array([1,1,1])
	for _ in range(iteration):
		p -= 2.0 * min(0.0, np.dot(p, reflect_normals[0])) * reflect_normals[0]
		p -= 2.0 * min(0.0, np.dot(p, reflect_normals[1])) * reflect_normals[1]
		p -= 2.0 * min(0.0, np.dot(p, reflect_normals[2])) * reflect_normals[2]
		p = p * scale - offset * scale
	p /= scale
	tet_norm = np.array([[-1,1,1],[1,-1,1],[1,1,-1],[-1,-1,-1]]) / np.sqrt(3)
	tet_offs = np.array([[1,1,1],[1,1,1],[1,1,1],[1,-1,-1]])
	tet_dist = ((p - tet_offs) * tet_norm).sum(axis=1).max()
	return tet_dist * pow(scale, float(-iteration+1))

@jit(nopython=True, nogil=True)
def DE_cube(p):
	p[2] += 3
	#p = rotation_matrix(0.6*p[0], 0, 0) @ p
	p = rotation_matrix(0.79, 0.7, 0.63) @ p
	iteration = 10
	scale = 3.0
	vtx = np.array([[-1,-1,-1],[-1,-1,0],[-1,-1,1],[-1,0,-1],[-1,0,1],[-1,1,-1],[-1,1,0],[-1,1,1],\
                    [0,-1,-1],[0,-1,1],[0,1,-1],[0,1,1],\
                    [1,-1,-1],[1,-1,0],[1,-1,1],[1,0,-1],[1,0,1],[1,1,-1],[1,1,0],[1,1,1]])
	for _ in range(iteration):
		v = np.square(vtx - p).sum(axis=1).argmin()
		p = p * scale - vtx[v] * scale
	p /= scale
	cube_norm = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
	cube_offs = np.array([[1,1,1],[-1,-1,-1],[1,1,1],[-1,-1,-1],[1,1,1],[-1,-1,-1]])
	cube_dist = ((p - cube_offs) * cube_norm).sum(axis=1).max()
	return cube_dist * pow(scale, float(-iteration+1))

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
	elif object_type == 'cube':
		DistanceEstimator = DE_cube
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