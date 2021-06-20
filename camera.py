import cupy as np
from cupy.linalg import norm as vecLen

from params import *

def getPixelData(width, height):
	ratio = height / width
	xCor = np.linspace(-1, 1, width)
	yCor = np.linspace(-ratio, ratio, height)

	pixelCoors = np.zeros((width, height, 3))
	pixelCoors[:,:,0] = xCor[:, np.newaxis]
	pixelCoors[:,:,1] = yCor
	pixelCoors[:,:,2] = 2

	direction = np.zeros((width, height, 3))
	direction = pixelCoors - CAM_POS
	direction = direction / vecLen(direction, axis=2)[:, :, np.newaxis]

	pixelCoors = np.resize(pixelCoors, (width * height, 3))
	direction  = np.resize(direction , (width * height, 3))

	return [pixelCoors, direction]
