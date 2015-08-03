"""
Provide some basic helper functions.
"""

import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping

def f2K(f, shape):
    """ Convert focal length to camera intrinsic matrix. """

    K = np.matrix([[f, 0, 0.5*shape[1]],
                   [0, f, 0.5*shape[0]],
                   [0, 0, 1]], dtype=np.float)

    return K

def E2Rt(E, K, baseRt, frameIdx, kp1, kp2, matches):
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
        matches1.append((pt1, m.queryIdx, frameIdx))

        pt2 = np.matrix([kp2[m.trainIdx].pt[1], kp2[m.trainIdx].pt[0], 1]).T
        matches2.append((pt2, m.trainIdx, frameIdx + 1))

    # create four possible new camera matrices
    Rt1 = np.hstack([R1, t1])
    Rt2 = np.hstack([R1, t2])
    Rt3 = np.hstack([R2, t1])
    Rt4 = np.hstack([R2, t2])

    # transform each Rt to be relative to baseRt
    baseRt4x4 = np.vstack([baseRt, np.matrix([0, 0, 0, 1], dtype=np.float)])
    Rt1 = Rt1 * baseRt4x4
    Rt2 = Rt2 * baseRt4x4
    Rt3 = Rt3 * baseRt4x4
    Rt4 = Rt4 * baseRt4x4

    # test how many points are in front of both cameras    
    bestRt = None
    bestCount = -1
    bestPts3D = None
    for Rt in [Rt1, Rt2, Rt3, Rt4]:

        cnt = 0
        pts3D = {}
        for m1, m2 in zip(matches1, matches2):

            # use least squares triangulation
            X = triangulateLS(baseRt, Rt, m1[0], m2[0], K)
            x = fromHomogenous(X)
            pts3D[x] = (m1, m2)

            # test if in front of both cameras
            if inFront(baseRt, x) and inFront(Rt, x):
                cnt += 1

        # update best camera/cnt
        if cnt > bestCount:
            bestCount = cnt
            bestRt = Rt
            bestPts3D = pts3D

    print "Found %d of %d possible 3D points in front of both cameras." % (bestCount, len(matches1))
    return bestRt, bestPts3D

def updateGraph(graph, Rt, pts3D):
    """ Update graph dictionary with new pose and 3D points. """

    # append new pose
    graph["motion"].append(Rt)

    # insert 3D points, checking for matches with existing points
    for X, matches in pts3D.iteritems():
        m1, m2 = matches
        oldKey = (m1[1], m1[2])
        newKey = (m2[1], m2[2])

        # if there's a match, update that entry
        if oldKey in graph["3Dmatches"]:
            entry = graph["3Dmatches"][oldKey]
            entry["frames"].append(m2[2])
            entry["2Dlocs"].append(m2[0])
            entry["3Dlocs"].append(X)

            del graph["3Dmatches"][oldKey]
            graph["3Dmatches"][newKey] = entry

        # otherwise, create new entry
        else:
            entry = {"frames" : [m1[2], m2[2]],
                     "2Dlocs" : [m1[0], m2[0]],
                     "3Dlocs" : [X]}
            graph["3Dmatches"][newKey] = entry

def finalizeGraph(graph):
    """ 
    Replace the 3Dlocs list with the its average for each entry.
    """

    pts3D = []
    for key, entry in graph["3Dmatches"].iteritems():
        
        # compute average
        tot = 0
        
        for X in entry["3Dlocs"]:
            tot += X

        avg = tot / len(entry["3Dlocs"])

        # append entry to list
        pts3D.append((avg, entry["frames"], entry["2Dlocs"]))

    # update
    graph["3Dmatches"] = pts3D

def bundleAdjustment(graph, K):
    """ Run bundle adjustment to joinly optimize camera poses and 3D points. """

    # unpack graph parameters into 1D array for initial guess
    num_frames = len(graph["motion"])
    x0, views, pts2D = unpackGraph(graph)

    # run basinhopping algorithm
    minimizer_kwargs = {"method" : "Nelder-Mead", "args" : (K, views, pts2D)}
    result = basinhopping(reprojectionError, x0, 
                          minimizer_kwargs=minimizer_kwargs, niter=200)

    # repack into graph
    optimized_pts3D = extract3DPts(result.x, num_frames)
    return optimized_graph

def unpackGraph(graph):
    """ Extract parameters for optimization. """

    # extract motion parameters, except initial pose
    motion = np.array(graph["motion"])[1:].ravel()

    # extract frame parameters as array for all frames at each point
    views = []
    pts3D = []
    pts2D = []

    for pt in graph["3Dmatches"]:
        views.append(pt[1])
        pts3D.append(pt[0])
        pts2D.append(pt[2])

    structure = np.array(pts3D).ravel()

    # concatenate arrays and return
    x0 = np.hstack(motion, structure)

    return x0, views, pts2D


def reprojectionError(x, K, views, pts2D):
    """ Compute reprojection error for the graph with these parameters. """

    """
    WRITE THIS!!!
    """

    pass

def toAxisAngle(R):
    """ 
    Decompose rotation R to axis-angle representation, where the angle,
    in radians, is given as the magnitude of the axis vector.
    """

    # extract 1-eigenvector
    U, V = np.linalg.eig(R)
    axis = np.array(np.real(V[:, np.where(U == 1)[0]]).T)[0]

    # try both possible angles
    angle1 = np.arccos(0.5 * (np.trace(R) - 1))
    angle2 = 2 * np.pi - angle1

    R1 = fromAxisAngle(axis * angle1)
    R2 = fromAxisAngle(axis * angle2)

    err1 = R - R1
    err2 = R - R2

    if np.multiply(err1, err1).sum() < np.multiply(err2, err2).sum():
        return axis * angle1
    return axis * angle2

def fromAxisAngle(r):
    """ Convert axis-angle representation to full rotation matrix. """

    # from https://en.wikipedia.org/wiki/Rotation_matrix
    angle = np.sqrt(np.multiply(r, r).sum())
    axis = r / angle

    cross = np.matrix([[0, -axis[2], axis[1]],
                       [axis[2], 0, -axis[0]],
                       [-axis[1], axis[0], 0]], dtype=np.float)
    tensor = np.matrix([[axis[0]**2, axis[0]*axis[1], axis[0]*axis[2]],
                        [axis[0]*axis[1], axis[1]**2, axis[1]*axis[2]],
                        [axis[0]*axis[2], axis[1]*axis[2], axis[2]**2]], dtype=np.float)

    R = np.cos(angle)*np.eye(3) + np.sin(angle)*cross + (1-np.cos(angle))*tensor
 
    return R

def extract3DPts(x, num_frames):
    """ Extract 3D points from parameter vector. """

    # only consider the entries of x representing 3D structure
    offset = (num_frames - 1) * 12
    structure = x[offset:]

    # repack structure into 3D points
    num_pts = len(structure) / 3
    pts3D = np.hsplit(np.matrix(np.split(structure, num_pts)).T, num_pts)

    return pts3D

def extractPoses(x, num_frames):
    """ Extract camera poses from parameter vector. """

    # only consider the entries of x representing poses
    offset = (num_frames - 1) * 12
    motion = x[:offset]

    # repack motion into 3x4 pose matrices
    pose_arrays = np.split(motion, num_frames - 1)
    pose_matrices = []
    for p in pose_arrays:
        pose_matrices.append(np.matrix(p.reshape((3, 4))))

    return pose_matrices

def printGraphStats(graph):
    """ Compute and display summary statistics for graph dictionary. """

    print "\nNumber of frames: " + str(len(graph["motion"]))
    print "Number of 3D points: " + str(len(graph["3Dmatches"].keys()))

    # count multiple correspondence
    cnt = 0
    for key, entry in graph["3Dmatches"].iteritems():
        if len(entry["frames"]) > 2:
            cnt += 1

    print "Number of 3D points with >1 correspondence(s): " + str(cnt)
    print ""

def inFront(P, X):
    """ Return true if X is in front of the camera. """

    R = P[:, :-1]
    t = P[:, -1]

    if R[2, :] * (X + R.T * t) > 0:
        return True
    return False 

def triangulateLS(Rt1, Rt2, x1, x2, K):
    """ 
    Triangulate a least squares 3D point given two camera matrices
    and the point correspondence in homogeneous coordinates.
    """
    
    A = np.vstack([K * Rt1, K * Rt2])
    b = np.vstack([x1, x2])
    X = np.linalg.lstsq(A, b)[0]

    return X

def fromHomogenous(X):
    """ Transform a point from homogenous to normal coordinates. """

    x = X[:-1]
    x /= X[-1]

    return x

def toHomogenous(x):
    """ Transform a point from normal to homogenous coordinates. """

    X = np.vstack([x, 1])

    return X

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