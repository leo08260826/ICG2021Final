import numpy as np
from numpy.linalg import norm as vecLen

from params import *

def getPixelData(width, height):
	ratio = height / width
	xCor = np.linspace(-1, 1, width)
	yCor = np.linspace(-ratio, ratio, height)

	pixelCoors = np.zeros((height, width, 3))
	pixelCoors[:,:,0] = yCor[:, np.newaxis]
	pixelCoors[:,:,1] = xCor
	pixelCoors[:,:,2] = 2

	direction = np.zeros((height, width, 3))
	direction = pixelCoors - CAM_POS
	direction = direction / vecLen(direction, axis=2)[:, :, np.newaxis]

	pixelCoors.resize((width * height, 3))
	direction .resize((width * height, 3))
	result = list(zip(list(pixelCoors), list(direction)))
	return result