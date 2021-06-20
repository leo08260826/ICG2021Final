import cupy as np
from cupy.linalg import norm

### constants
UX = np.array([1.0, 0.0, 0.0])
UY = np.array([0.0, 1.0, 0.0])
UZ = np.array([0.0, 0.0, 1.0])
O = np.array([0.0, 0.0, 0.0])
I = np.array([1.0, 1.0, 1.0])
WHITE = np.array([255.0, 255.0, 255.0])

### utilities
def unit(vec): return vec / norm(vec, axis=-1)[:, np.newaxis]
