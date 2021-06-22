import numpy as np
from numpy.linalg import norm as vecLen

from params import *
from util import *

def getPixelData(width, height, rotation=I):
	ratio = height / width
	xCor = np.linspace(-1, 1, width)
	zCor = np.linspace(ratio, -ratio, height)

	pixelCoors = np.zeros((height, width, 3))
	pixelCoors[:,:,0] = xCor
	pixelCoors[:,:,1] = 2
	pixelCoors[:,:,2] = zCor[:, np.newaxis]

	directions = pixelCoors / norm(pixelCoors, axis=-1)[..., np.newaxis]
	directions = np.einsum('ij,lkj->lki', rotation, directions)
	directions.resize((width * height, 3))

	cam_pos = np.tile(CAM_POS, width * height)
	cam_pos.resize((width * height, 3))

	result = list(zip(list(cam_pos), list(directions)))

	return result
