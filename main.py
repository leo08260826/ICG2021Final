import multiprocessing

from datetime import datetime as Time

from numpy import array
import cv2

from camera import getPixelData
from raymarch import rayMarching

from params import *

def strTime(time):
	return str(time.total_seconds()) + ' s'

if __name__ == '__main__':

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

	### ray marching
	starttime = Time.now()
	results = pool.starmap(rayMarching, pixelData)
	endtime = Time.now()	
	time = endtime - starttime
	print("Ray Marching done in " + strTime(time) + " .")
	print("Average time per 10000 pixels : " +
		strTime(time / WIDTH / HEIGHT * 10000))
	print("Average time per 10000 rays   : " +
		strTime(time / WIDTH / HEIGHT / RAYS_SCALE / RAYS_SCALE * 10000))

	### image showing / storing
	image = array(results).reshape(HEIGHT * RAYS_SCALE, WIDTH * RAYS_SCALE, 3)
	if RAYS_SCALE != 1:
		image = cv2.resize(image, dsize=(WIDTH, HEIGHT))
	# cv2.imshow('Fractal Rendering', image / 255)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	cv2.imwrite('output.jpg', image)
