from numpy import array
from numpy import linspace
from numpy.linalg import norm
from numba import jit

from params import *
from util import *

@jit(nopython=True, nogil=True)
def getPixelData(width, height):
	xCor = linspace(-1, 1, width)
	yCor = linspace(-1*height/width, 1*height/width, height)
	zCor = 2

	data = []
	for x in xCor:
		for y in yCor:
			pixelCor = array([x, y, zCor])
			direction = unit(pixelCor - CAM_POS)
			data.append((pixelCor, direction))

	return data