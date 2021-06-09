import pygame
import numpy as np

### para of settings
SCREEN_HEIGHT = 300
SCREEN_WIDTH = 400
FPS = 12
cameraCor = np.array([0, 0, 4])

### parameter of ray marching
MaximumRaySteps = 10
MinimumDistance = 0.01

def normalize(vector):
	return vector/np.linalg.norm(vector)

def DistanceEstimator(point):
	point[0] = point[0]%1.0 - 0.5
	point[1] = point[1]%1.0 - 0.5
	return np.linalg.norm(point)-0.3

def rayTracing(pixel, direction): # actually ray marching
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

if __name__ == '__main__':
	### init pygame
	pygame.init()
	display_surface = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
	pygame.display.set_caption("Fractal Rendering")

	### init data of every pixel of image
	xCor = np.linspace(-1, 1, SCREEN_WIDTH)
	yCor = np.linspace(-1*SCREEN_HEIGHT/SCREEN_WIDTH, 1*SCREEN_HEIGHT/SCREEN_WIDTH, SCREEN_HEIGHT)
	zCor = 2

	pixelCor = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT, 3))
	for i in range(SCREEN_WIDTH):
		for j in range(SCREEN_HEIGHT):
			pixelCor[i][j] = np.array([xCor[i], yCor[j], zCor])
			
	direction = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT, 3))
	for i in range(SCREEN_WIDTH):
		for j in range(SCREEN_HEIGHT):
			direction[i][j] = normalize(pixelCor[i][j] - cameraCor);

	image = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT, 3))

	### start game loop
	clock = pygame.time.Clock()
	flag = True # for now, only render once
	while True:
		clock.tick(FPS)

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				quit()

		if flag == True:
			for i in range(SCREEN_WIDTH):
				for j in range(SCREEN_HEIGHT):
					image[i][j] = rayTracing(pixelCor[i][j], direction[i][j])
		
			imageSurf = pygame.surfarray.make_surface(image)
			display_surface.blit(imageSurf, (0, 0))
			pygame.display.update()

			flag=False

		