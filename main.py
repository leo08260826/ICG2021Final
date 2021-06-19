import cv2
import numpy as np
import camera
from raymarch import rayMarching
import multiprocessing
import datetime

### para of settings
SCREEN_HEIGHT = 300
SCREEN_WIDTH = 400

if __name__ == '__main__':
	### init multiprocessing
	cpus = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(cpus)
	print("CPU nums: " + str(cpus))

	### init data of every pixel of image
	starttime = datetime.datetime.now()
	pixelData = camera.getPixelData(SCREEN_HEIGHT, SCREEN_WIDTH)
	endtime = datetime.datetime.now()
	print("finish init. (time: " + str((endtime - starttime).seconds) + "s)")

	### rendering
	starttime = datetime.datetime.now()
	results = pool.starmap(rayMarching, pixelData)
	image = np.array(results).reshape(SCREEN_HEIGHT, SCREEN_WIDTH, 3)
	endtime = datetime.datetime.now()	
	print("finish rendering image. (time: " + str((endtime - starttime).seconds)+ "s)")

	### image showing/ storing
	cv2.imshow('Fractal Rendering', image/255)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	# cv2.imwrite('output.jpg', image)

	



		