import numpy as np

WIDTH = 480
HEIGHT = 270
RAYS_SCALE = 1 # anti-aliasing

FRACTAL = 'tetrahedron_with_floor'
MAX_STEP = 256
MIN_DIST = 1 / WIDTH
ITER = 3

CAM_POS = np.array([-10.0, 0.0, 2.0])
CAM_ANG = np.array([0.0, 0.0, 0.0])

L_POS = np.array([-10.0, -10.0, 10.0])
L_ITEN = 250.0

AMB_I = 0.1
AMB_R = 0.1
DIF_R = 1.0
SPC_R = 0.2
SPC_P = 8.0

REF_R = 0.5
SSH_S = 16.0 # shadow softness
SSH_I = 0.4
