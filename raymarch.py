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
def DE_tetrahedron(point):
	point = rotation_matrix(0.48, 0.7, 0.29) @ point
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

@jit(nopython=True, nogil=True)
def DE_cube(p):
	p = rotation_matrix(0.79, 0.7, 0.63) @ p
# 	iteration = 10
# 	scale = 3.0
# 	nor = np.array([[1.0,0,0],[0,1.0,0],[0,0,1.0],[1.0,0,0],[0,1.0,0],[0,0,1.0]])
# 	off = np.array([[1/3,0,0],[0,1/3,0],[0,0,1/3],[0.0,0,0],[0,0.0,0],[0,0,0.0]])
# 	init = np.array([1,1,1])
# 	for _ in range(iteration):
# 		p -= 2.0 * min(0.0, np.dot(p - off[3], nor[3])) * nor[3] + off[3]
# 		p -= 2.0 * min(0.0, np.dot(p - off[4], nor[4])) * nor[4] + off[4]
# 		p -= 2.0 * min(0.0, np.dot(p - off[5], nor[5])) * nor[5] + off[5]
# 		dot0 = np.dot(p - off[0], nor[0])
# 		dot1 = np.dot(p - off[1], nor[1])
# 		dot2 = np.dot(p - off[2], nor[2])
# 		if dot0 < 0 and dot1 >= 0 and dot2 >= 0:
# 			p -= 2.0 * dot0 * nor[0]
# 		if dot1 < 0 and dot0 >= 0 and dot2 >= 0:
# 			p -= 2.0 * dot1 * nor[1]
# 		if dot2 < 0 and dot1 >= 0 and dot0 >= 0:
# 			p -= 2.0 * dot2 * nor[2]
# 		p = p * scale - init * (scale - 1.0)
# 	return np.linalg.norm(p) * pow(scale, float(-iteration))
	iteration = 10
	scale = 3.0
	vtx = np.array([[-1,-1,-1],[-1,-1,0],[-1,-1,1],[-1,0,-1],[-1,0,1],[-1,1,-1],[-1,1,0],[-1,1,1],\
                    [0,-1,-1],[0,-1,1],[0,1,-1],[0,1,1],\
                    [1,-1,-1],[1,-1,0],[1,-1,1],[1,0,-1],[1,0,1],[1,1,-1],[1,1,0],[1,1,1]])/1.5
	for _ in range(iteration):
		v = np.square(vtx - p).sum(axis=1).argmin()
		p = p * scale - vtx[v] * (scale - 0.0)
	return np.linalg.norm(p) * pow(scale, float(-iteration))

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