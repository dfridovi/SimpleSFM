"""
Provide some basic helper functions.
"""

import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def f2K(f, shape):
    """ Convert focal length to camera intrinsic matrix. """

    K = np.matrix([[f, 0, 0.5*shape[1]],
                   [0, f, 0.5*shape[0]],
                   [0, 0, 1]], dtype=np.float)

    return K

def E2Rt(E, kp1, kp2, matches):
    """ Convert essential matrix to pose. From H&Z p.258. """

    W = np.matrix([[0, -1, 0],
                   [1, 0, 0],
                   [0, 0, 1]], dtype=np.float)
#    Z = np.matrix([[0, 1, 0],
#                   [-1, 0, 0],
#                   [0, 0, 0]], dtype=np.float)

    U, D, V = np.linalg.svd(E)

    # skew-symmetric translation matrix 
#    S = U * Z * U.T

    # two possibilities for R, t
    R1 = U * W * V.T
    R2 = U * W.T * V.T

    t1 = U[:, 2]
    t2 = -U[:, 2]

    # ensure positive determinants
    if np.linalg.det(R1) < 0:
        R1 = -R1

    if np.linalg.det(R2) < 0:
        R2 = -R2

    # extract match points
    matches1 = []
    matches2 = []
    for m in matches:
        pt1 = np.matrix([kp1[m.queryIdx].pt[1], kp1[m.queryIdx].pt[0], 1]).T
        matches1.append(pt1)

        pt2 = np.matrix([kp2[m.trainIdx].pt[1], kp2[m.trainIdx].pt[0], 1]).T
        matches2.append(pt2)

    # create four possible new camera matrices and original
    Rt0 = np.matrix(np.hstack([np.eye(3), np.zeros((3, 1))]))
    Rt1 = np.hstack([R1, t1])
    Rt2 = np.hstack([R1, t2])
    Rt3 = np.hstack([R2, t1])
    Rt4 = np.hstack([R2, t2])

    # test how many points are in front of both cameras    
    Rtbest = None
    bestCount = -1
    for Rt in [Rt1, Rt2, Rt3, Rt4]:

        cnt = 0
        for m1, m2 in zip(matches1, matches2):

            # use least squares triangulation
            X = triangulateLS(Rt0, Rt, m1, m2)
            x = fromHomogenous(X)

            # test if in front of both cameras
            if inFront(Rt0, x) and inFront(Rt, x):
                cnt += 1

        # update best camera/cnt
        if cnt > bestCount:
            bestCount = cnt
            Rtbest = Rt

    return Rtbest

def inFront(P, X):
    """ Return true if X is in front of the camera. """

    R = P[:, :-1]
    t = P[:, -1]

    if R[2, :] * (X + R.T * t) > 0:
        return True
    return False 

def triangulateLS(P1, P2, x1, x2):
    """ 
    Triangulate a least squares 3D point given two camera matrices
    and the point correspondence in homogeneous coordinates.
    """
    
    A = np.vstack([P1, P2])
    b = np.vstack([x1, x2])
    X = np.linalg.lstsq(A, b)[0]

    return X

def fromHomogenous(X):
    """ Transform a point from homogenous to normal coordinates. """

    x = X[:-1]
    x /= X[-1]

    return x

def ij2xy(i, j, shape):
    """ Convert array indices to xy coordinates. """

    x = j - 0.5*shape[1]
    y = 0.5*shape[0] - i

    return (x, y)

def drawMatches(img1, kp1, img2, kp2, matches):
    """ Visualize keypoint matches. """

    # get dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # create display
    view = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)

    view[:h1, :w1, :] = img1
    view[:h2, w2:, :] = img2

    color = (0, 255, 0)

    for idx, m in enumerate(matches):
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