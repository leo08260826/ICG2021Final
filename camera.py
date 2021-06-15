from numpy import array
from numpy import linspace
from numpy import zeros
from numpy.linalg import norm
from numba import jit

CameraCor = array([0, 0, 4])

@jit(nopython=True, nogil=True)
def normalize(vector):
	return vector/norm(vector)

@jit(nopython=True, nogil=True)
def getPixelData(width, height):
	xCor = linspace(-1, 1, width)
	yCor = linspace(-1*height/width, 1*height/width, height)
	zCor = 2

	cameraCor = CameraCor

	data = []
	append = data.append

	for x in xCor:
		for y in yCor:
			pixelCor = array([x, y, zCor])
			direction = normalize(pixelCor - cameraCor)
			append((x,y,zCor,direction[0],direction[1],direction[2]))

	return data