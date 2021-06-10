import numpy as np

cameraCor = np.array([0, 0, 4])

def normalize(vector):
	return vector/np.linalg.norm(vector)

def getPixelData(width, height):
	xCor = np.linspace(-1, 1, width)
	yCor = np.linspace(-1*height/width, 1*height/width, height)
	zCor = 2

	pixelCor = np.zeros((width, height, 3))
	for i in range(width):
		for j in range(height):
			pixelCor[i][j] = np.array([xCor[i], yCor[j], zCor])
			
	direction = np.zeros((width, height, 3))
	for i in range(width):
		for j in range(height):
			direction[i][j] = normalize(pixelCor[i][j] - cameraCor);

	return pixelCor, direction