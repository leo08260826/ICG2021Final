import numpy as np

WIDTH = 1280
HEIGHT = 720
RAYS_SCALE = 1 # anti-aliasing
FPS = 5

MAX_STEP = 256
MIN_DIST = 0.001

CAM_POS = np.array([0.0, 0.0, 4.0])

L_POS = np.array([10.0, 10.0, 10.0])
L_ITEN = 250.0

AMB_I = 0.1
AMB_R = 0.05
DIF_R = 1.0
SPC_R = 0.2
SPC_P = 8.0
