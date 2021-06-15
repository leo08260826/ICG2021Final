import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np
import camera
from raymarch import rayMarching
import multiprocessing

import datetime

### para of settings
SCREEN_HEIGHT = 300
SCREEN_WIDTH = 400
FPS = 12

if __name__ == '__main__':
	### init pygame
	pygame.init()
	display_surface = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
	pygame.display.set_caption("Fractal Rendering")

	### init data of every pixel of image
	pixelData = camera.getPixelData(SCREEN_WIDTH, SCREEN_HEIGHT)

	### init multiprocessing
	cpus = multiprocessing.cpu_count()
	print("CPU nums: " + str(cpus))
	pool = multiprocessing.Pool(cpus)

	### start game loop
	clock = pygame.time.Clock()
	print("FPS: " + str(FPS))
	flag = True # for now, only render once
	starttime = datetime.datetime.now() # counting time
	while True:
		clock.tick(FPS)

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				quit()

		if flag == True:
			# for i in range(SCREEN_WIDTH):
			# 	for j in range(SCREEN_HEIGHT):
			# 		image[i][j] = rayMarching(pixelCor[i][j], direction[i][j])
			
			results = pool.starmap(rayMarching, pixelData)
			image = np.array(results)
			image = image.reshape(SCREEN_WIDTH, SCREEN_HEIGHT, 3)

			imageSurf = pygame.surfarray.make_surface(image)
			display_surface.blit(imageSurf, (0, 0))
			pygame.display.update()

			flag=False
			endtime = datetime.datetime.now()
			print("time: " + str((endtime - starttime).seconds))

		