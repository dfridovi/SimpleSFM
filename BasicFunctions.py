"""
Provide some basic helper functions.
"""

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def f2K(f):
	""" Convert focal length to camera intrinsic matrix. """

	K = np.matrix([[f, 0, 0],
				   [0, f, 0],
				   [0, 0, 1]], dtype=np.float)

def drawMatches(img1, kp1, img2, kp2, matches):
	""" Visualize keypoint matches. """

	"""
	WORK ON THIS!
	"""

	pass

def imread(imfile):
    """ Read image from file and normalize. """

    img = mpimg.imread(imfile).astype(np.float)
    img = rescale(img)
    return img

def imshow(img, title="", cmap="gray", cbar=False):
    """ Show image to screen. """
    
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    
    if cbar:
        plt.colorbar()

    plt.show()

def imsave(img, imfile):
    """ Save image to file."""

    mpimg.imsave(imfile, img)

def rescale(img):
    """ Rescale image values linearly to the range [0.0, 1.0]. """

    return (img - img.min()) / (img.max() - img.min())

def truncate(img):
    """ Truncate values in image to range [0.0, 1.0]. """

    img[img > 1.0] = 1.0
    img[img < 0.0] = 0.0
    return img

def rgb2gray(img):
    """ Convert an RGB image to grayscale. """

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    return 0.299*r + 0.587*g + 0.114*b