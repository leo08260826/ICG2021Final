import pygame
import numpy as np
import camera
from raymarch import rayMarching

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
	pixelCor, direction = camera.getPixelData(SCREEN_WIDTH, SCREEN_HEIGHT)

	image = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT, 3))

	### start game loop
	clock = pygame.time.Clock()
	flag = True # for now, only render once
	starttime = datetime.datetime.now() # counting time
	while True:
		clock.tick(FPS)

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				quit()

		if flag == True:
			for i in range(SCREEN_WIDTH):
				for j in range(SCREEN_HEIGHT):
					image[i][j] = rayMarching(pixelCor[i][j], direction[i][j])
		
			imageSurf = pygame.surfarray.make_surface(image)
			display_surface.blit(imageSurf, (0, 0))
			pygame.display.update()

			flag=False
			endtime = datetime.datetime.now()
			print((endtime - starttime).seconds)

		