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
FPS = 5

if __name__ == '__main__':
	### init pygame
	pygame.init()
	display_surface = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
	pygame.display.set_caption("Fractal Rendering")

	### init multiprocessing
	cpus = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(cpus)
	print("CPU nums: " + str(cpus))
	print("FPS: " + str(FPS))

	### init data of every pixel of image
	starttime = datetime.datetime.now()
	pixelData = camera.getPixelData(SCREEN_WIDTH, SCREEN_HEIGHT)
	endtime = datetime.datetime.now()
	print("finish init. (time: " + str((endtime - starttime).seconds) + "s)")

	### start game loop
	clock = pygame.time.Clock()
	flag = True # for now, only render once
	while True:
		
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				quit()

		if flag == True:
			starttime = datetime.datetime.now() # counting time	
			results = pool.starmap(rayMarching, pixelData)
			image = np.array(results).reshape(SCREEN_WIDTH, SCREEN_HEIGHT, 3)

			imageSurf = pygame.surfarray.make_surface(image)
			display_surface.blit(imageSurf, (0, 0))
			pygame.display.update()

			flag=False
			endtime = datetime.datetime.now()	
			print("image time: " + str((endtime - starttime).seconds))

		clock.tick(FPS)

		