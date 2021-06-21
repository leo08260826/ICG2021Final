import numpy as np

WIDTH = 480
HEIGHT = 270
RAYS_SCALE = 1 # anti-aliasing

MAX_STEP = 256
MIN_DIST = 1 / WIDTH

CAM_POS = np.array([0.0, -10.0, 1.0])

L_POS = np.array([10.0, -10.0, 10.0])
L_ITEN = 250.0

AMB_I = 0.1
AMB_R = 0.1
DIF_R = 1.0
SPC_R = 0.2
SPC_P = 8.0
