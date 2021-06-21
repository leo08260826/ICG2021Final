import numpy as np
from utils import normalize, rotation_matrix

def getPixelData(width, height, rotation = np.identity(3), translation = np.array([0,0,0])):
	cameraCor = np.array([0.0, 0, 0])
    
	xCor = -2
	yCor = np.linspace(-1, 1, width)
	zCor = np.linspace(-1*height/width, 1*height/width, height)

	pixelCor = np.zeros((height, width, 3))
	for i in range(height):
		for j in range(width):
			pixelCor[i][j] = np.array([xCor, yCor[j], zCor[height-1-i]])
			
	direction = np.zeros((height, width,3))
	for i in range(height):
		for j in range(width):
			direction[i][j] = rotation @ normalize(pixelCor[i][j] - cameraCor);

	cameraCor = rotation @ cameraCor
	cameraCor += translation
	return cameraCor , direction