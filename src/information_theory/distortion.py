'''Distortion computation.'''

from .information import average_energy
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim

def MSE(x, y):
    error_signal = x.astype(np.float64) - y
    return average_energy(error_signal)

def RMSE(x, y):
    return math.sqrt(MSE(x, y))

# Set channel_axis=0 for grayscale images.
def SSIM(x, y, channel_axis=2):
    return ssim(x, y, data_range=y.max() - y.min(), full=False, channel_axis=channel_axis)
