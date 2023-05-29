"""
CLAB Task-1: Harris Corner Detector
Your name (Your uniID):
"""

import numpy as np


def conv2(img, conv_filter):
    # flip the filter
    f_siz_1, f_size_2 = conv_filter.shape
    conv_filter = conv_filter[range(f_siz_1 - 1, -1, -1), :][:, range(f_siz_1 - 1, -1, -1)]
    pad = (conv_filter.shape[0] - 1) // 2
    result = np.zeros((img.shape))
    img = np.pad(img, ((pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))
    filter_size = conv_filter.shape[0]
    for r in np.arange(img.shape[0] - filter_size + 1):
        for c in np.arange(img.shape[1] - filter_size + 1):
            curr_region = img[r:r + filter_size, c:c + filter_size]
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result)  # Summing the result of multiplication.
            result[r, c] = conv_sum  # Saving the summation in the convolution layer feature map.

    return result


def fspecial(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


# Parameters, add more if needed
sigma = 2
thresh = 0.01
#range of k should be 0.01- 0.1
k = 0.05  

# Derivative masks
dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
dy = dx.transpose()
import matplotlib.pyplot as plt

bw = plt.imread('0.png')
bw = np.array(bw * 255, dtype=int)
# computer x and y derivatives of image
Ix = conv2(bw, dx)
Iy = conv2(bw, dy)

# Set up a gaussian kernel with shape of no less than 1*1, the shape is dependent on the sigma size 
g = fspecial((max(1, np.floor(3 * sigma) * 2 + 1), max(1, np.floor(3 * sigma) * 2 + 1)), sigma)
Iy2 = conv2(np.power(Iy, 2), g)
Ix2 = conv2(np.power(Ix, 2), g)
Ixy = conv2(Ix * Iy, g)

######################################################################
# Task: Compute the Harris Cornerness
######################################################################


######################################################################
# Task: Perform non-maximum suppression and
#       thresholding, return the N corner points
#       as an Nx2 matrix of x and y coordinates
######################################################################

######################################################################
# Task: Compute the Harris Cornerness
######################################################################


kernel = np.ones((5,5))
lambda_1 = conv2(Ix2, kernel)
lambda_2 = conv2(Iy2, kernel)
squared_sum_Ixy =  np.square(conv2(Ixy, kernel))

# calculate determinant and trace of matrix M for each pixel in the image, and finally compute R
det_m = lambda_1 * lambda_2 - squared_sum_Ixy
trace_m = lambda_1 + lambda_2
R = det_m - (k * trace_m**2)

######################################################################
# Task: Perform non-maximum suppression and
#       thresholding, return the N corner points
#       as an Nx2 matrix of x and y coordinates
######################################################################

def non_maximum_suppression(R):
    """
    Perform non-maximum suppression by finding pixels that have intensity greater than 8 neighbors.
    
    input:
        R: 2-d numpy array of corner response values
    returns:
        corners: list of tuples indicating corners ditected

        arr: 2-d numpy binary array where arr[i, j] == 1 indicates corner as detected by non-maximum suppression
    """
    
    corners = []
    for i in range(1, len(R)-1):
        for j in range(1, len(R[0])-1):
            # check if R[i,j] is a corner by comparing each pixel to 8 neighbors
            if ((R[i,j] >= R[i+1,j]) & (R[i,j] >= R[i-1,j]) & (R[i,j] >= R[i,j+1]) & (R[i,j] >= R[i,j-1]) & 
                (R[i,j] >= R[i+1,j+1]) & (R[i,j] >= R[i-1,j-1]) & (R[i,j] >= R[i-1,j+1]) & (R[i,j] >= R[i-1,j-1])):
                    if R[i,j] >= thresh:
                        corners += (i, j)
    # create a binary array from corners array to visualize corners
    arr = np.zeros(R.shape)
    for corner in corners:
        arr[corner[0], corner[1]] = 1
    return corners, arr

corners, Nx2 = non_maximum_suppression(R)

