import numpy as np
import distance_estimator
from numba import jit

### parameter of ray marching
MaximumRaySteps = 200
MinimumDistance = 0.001

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

@jit(nopython=True, nogil=True)
def normal(p, DE):
	dp = np.array([[MinimumDistance/10,0,0],[0,MinimumDistance/10,0],[0,0,MinimumDistance/10]])
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
	steps_inter = steps + distance / MinimumDistance
	occ_factor = pow(2, -float(steps_inter)/float(MaximumRaySteps - 1) / 0.1)
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
	totalDistance = MinimumDistance # avoid division by zero
	init += totalDistance * direction
	steps = 0
	illum = 1.0
	for steps in range(0, MaximumRaySteps):
		p = init + totalDistance * direction
		distance, c_obj = DE(p)
		illum = max(0.0, min(illum, shadow_softness * distance / totalDistance))
		totalDistance += distance
		if(distance < MinimumDistance):
			break
	return illum
            
@jit(nopython=True, nogil=True)
def rayMarching(camaraCor, direction, DE, reflect_count):
	totalDistance = 0.0
	steps = 0
	for steps in range(0, MaximumRaySteps):
		p = camaraCor + totalDistance * direction
		distance, c_obj = DE(p)
		totalDistance += distance
		if(distance < MinimumDistance):
			break
    
	if steps >= MaximumRaySteps-1:
		color = background_color(p)
	else:
		N = normal(p, DE)
		N /= np.linalg.norm(N)

		c_ambient = ambient(c_obj)       
		c_diffuse = diffuse(c_obj, N, p)
		c_specular = specular(N, p, -direction)
        
		f_occlusion = occlusion(c_obj, steps, distance) * w_occlusion + (1-w_occlusion)
		f_softshadow = softshadow(p, DE) * w_softshadow + (1-w_softshadow)

		color = np.array([0.0, 0.0, 0.0])
		color += w_ambient * c_ambient * f_occlusion
		color += w_diffuse * c_diffuse * f_softshadow
		color += w_specular * c_specular * f_softshadow

		if reflect_count > 0 and np.dot(N, -direction) > 0:
			reflect_direct = reflect_direction(N, -direction)
			reflect_p = p + reflect_direct * MinimumDistance
			reflect_color = rayMarching(reflect_p, reflect_direct, DE, reflect_count-1)
			color = (1-reflect_decay) * color + reflect_decay * reflect_color
	return color