import numpy as np
from numpy.linalg import norm

cameraCor = np.array([0, 0, 4])

def normalize(vector):
	return vector/norm(vector)

def getPixelData(width, height):
	xCor = np.linspace(-1, 1, width)
	yCor = np.linspace(-1*height/width, 1*height/width, height)
	zCor = 2

	data = []

	pixelCor = np.zeros((width, height, 3))
	for i in range(width):
		for j in range(height):
			pixelCor = np.array([xCor[i], yCor[j], zCor])
			direction = normalize(pixelCor - cameraCor)
			data.append((pixelCor[0],pixelCor[1],pixelCor[2],direction[0],direction[1],direction[2]))
	if(i==0 and j==0):
		print(data)

	return data