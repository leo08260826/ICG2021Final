import numpy as np
import distance_estimator as de

WIDTH = 3840
HEIGHT = 2160
RAYS_SCALE = 4 # anti-aliasing

FRACTAL = de.tetrahedron_with_floor
MAX_STEP = 1024
MIN_DIST = 1 / WIDTH
ITER = 4

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
