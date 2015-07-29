"""
Provide some basic helper functions.
"""

import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def f2K(f):
	""" Convert focal length to camera intrinsic matrix. """

	K = np.matrix([[f, 0, 0],
				   [0, f, 0],
				   [0, 0, 1]], dtype=np.float)

def E2pose(E):
    """ Convert essential matrix to pose. From H&Z p.258"""

    W = np.matrix([[0, -1, 0],
                   [1, 0, 0],
                   [0, 0, 1]], dtype=np.float)
    Z = np.matrix([[0, 1, 0],
                   [-1, 0, 0],
                   [0, 0, 0]], dtype=np.float)

    U, D, V = np.linalg.svd(E)

    # skew-symmetric translation matrix 
    #S = U * Z * U.T

    # two possibilities for R, t
    R1 = U * W * V.T
    R2 = U * W.T * V.T

    t1 = U[:, 2]
    t2 = -U[:, 2]

    # ensure positive determinants
    if np.linalg.det(R1) < 0
        R1 = -R1

    if np.linalg.det(R2) < 0
        R2 = -R2



def drawMatches(img1, kp1, img2, kp2, matches, matchesMask):
    """ Visualize keypoint matches. """

    # get dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # create display
    view = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)

    view[:h1, :w1, :] = img1
    view[:h2, w2:, :] = img2

    color = (0, 255, 0)

    if matchesMask is None:
        return

    for idx, m in enumerate(matches):
        if matchesMask[idx] == 1:
            cv2.line(view, 
                     (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])), 
                     (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1])), 
                     color)

    cv2.imshow("Keypoint correspondences", view)
    cv2.waitKey()
    cv2.destroyAllWindows()

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