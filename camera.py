import numpy as np
from numpy.linalg import norm as vecLen

from params import *
from util import *

def getPixelData(width, height, rotation = np.identity(3), translation = O):
	ratio = height / width
	xCor = np.linspace(-1, 1, width)
	yCor = np.linspace(-ratio, ratio, height)

	pixelCoors = np.zeros((height, width, 3))
	pixelCoors[:,:,0] = yCor[:, np.newaxis]
	pixelCoors[:,:,1] = xCor
	pixelCoors[:,:,2] = 2

	# directions = np.zeros((height, width, 3))
	directions = pixelCoors - CAM_POS
	directions = directions / norm(directions, axis=-1)[..., np.newaxis]
	directions = np.einsum('ij,lkj->lki', rotation, directions)
	directions.resize((width * height, 3))

	cam_pos = rotation @ CAM_POS
	cam_pos += translation
	cam_pos = np.repeat(cam_pos, width * height)
	cam_pos.resize((width * height, 3))

	result = list(zip(list(cam_pos), list(directions)))

	return result
