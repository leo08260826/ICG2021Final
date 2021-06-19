import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import multiprocessing

from datetime import datetime as Time

from numpy import array
import pygame
import cv2

from camera import getPixelData
from raymarch import rayMarching

from params import *

def strTime(time):
	return str(time.total_seconds()) + ' s'

if __name__ == '__main__':
	### init pygame
	pygame.init()
	display_surface = pygame.display.set_mode((WIDTH, HEIGHT))
	pygame.display.set_caption("Fractal Rendering")

	### init multiprocessing
	cpus = multiprocessing.cpu_count()
	used_cpus = cpus // 2 + 1
	pool = multiprocessing.Pool(used_cpus)
	print()
	print("Found " + str(cpus) + " CPU core(s).")
	print("Using " + str(used_cpus) + " CPU core(s).")
	print("FPS: " + str(FPS))

	### init data of every pixel of image
	startTime = Time.now()
	pixelData = getPixelData(WIDTH * RAYS_SCALE, HEIGHT * RAYS_SCALE)
	endTime = Time.now()
	print()
	print("Camera Initialized in " + strTime(endTime - startTime) + " .")

	### start game loop
	clock = pygame.time.Clock()
	flag = True # for now, only render once
	while True:
		
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				quit()

		if flag == True:
			starttime = Time.now() # counting time	
			results = pool.starmap(rayMarching, pixelData)
			image = array(results).reshape(WIDTH * RAYS_SCALE, HEIGHT * RAYS_SCALE, 3)
			if RAYS_SCALE != 1:
				image = cv2.resize(image, dsize=(HEIGHT, WIDTH))
			endtime = Time.now()	
			time = endtime - starttime
			print("Ray Marching done in " + strTime(time) + " .")
			print("Average time per 10000 pixels : " +
				strTime(time / WIDTH / HEIGHT * 10000))
			print("Average time per 10000 rays   : " +
				strTime(time / WIDTH / HEIGHT / RAYS_SCALE / RAYS_SCALE * 10000))

			imageSurf = pygame.surfarray.make_surface(image)
			display_surface.blit(imageSurf, (0, 0))
			pygame.display.update()

			flag=False

		clock.tick(FPS)

		