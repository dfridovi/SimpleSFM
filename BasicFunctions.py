"""
Provide some basic helper functions.
"""

import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import leastsq
from Queue import PriorityQueue

NUM_EVALS = 0
LAST_RMS_ERROR = 0.0

def f2K(f):
    """ Convert focal length to camera intrinsic matrix. """

    K = np.matrix([[f, 0, 0],
                   [0, f, 0],
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
        pt1 = np.matrix([kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]]).T
        matches1.append((pt1, m.queryIdx, frameIdx))

        pt2 = np.matrix([kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]]).T
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
            x = triangulateLM(baseRt, Rt, m1[0], m2[0], K)

            # test if in front of both cameras
            if inFront(baseRt, x) and inFront(Rt, x):
                cnt += 1
                pts3D[x] = (m1, m2)
    
        # update best camera/cnt
        #print "[DEBUG] Found %d points in front of both cameras." % cnt
        if cnt > bestCount:
            bestCount = cnt
            bestRt = Rt
            bestPts3D = pts3D

    print "Found %d of %d possible 3D points in front of both cameras.\n" % (bestCount, len(matches1))

    # Wrap bestRt, bestPts3D into a 'pair'
    pair = {}
    pair["motion"] = [baseRt, bestRt]
    pair["3Dmatches"] = {}
    pair["frameOffset"] = frameIdx
    for X, matches in bestPts3D.iteritems():
        m1, m2 = matches
        key = (m1[1], m1[2]) # use m1 instead of m2 for matching later

        entry = {"frames" : [m1[2], m2[2]], # frames that can see this point
                 "2Dlocs" : [m1[0], m2[0]], # corresponding 2D points
                 "3Dlocs" : X,              # 3D triangulation 
                 "newKey" : (m2[1], m2[2])} # next key (for merging with graph)
        pair["3Dmatches"][key] = entry
        
    return pair

def updateGraph(graph, pair):
    """ Update graph dictionary with new pose and 3D points. """

    # append new pose
    graph["motion"].append(pair["motion"][1])

    # insert 3D points, checking for matches with existing points
    for key, pair_entry in pair["3Dmatches"].iteritems():
        newKey = pair_entry["newKey"]

        # if there's a match, update that entry
        if key in graph["3Dmatches"]:
            graph_entry = graph["3Dmatches"][key]
            graph_entry["frames"].append(pair_entry["frames"][1])
            graph_entry["2Dlocs"].append(pair_entry["2Dlocs"][1])
            graph_entry["3Dlocs"].append(pair_entry["3Dlocs"])

            del graph["3Dmatches"][key]
            graph["3Dmatches"][newKey] = graph_entry

        # otherwise, create new entry
        else:
            graph_entry = {"frames" : pair_entry["frames"], # frames that can see this point
                           "2Dlocs" : pair_entry["2Dlocs"], # corresponding 2D points
                           "3Dlocs" : [pair_entry["3Dlocs"]], # 3D triangulations
                           "color"  : None}                  # point color
            graph["3Dmatches"][newKey] = graph_entry

def finalizeGraph(graph, frames):
    """ Replace the 3Dlocs list with the its average for each entry. Add color. """

    for key, entry in graph["3Dmatches"].iteritems():
        
        # compute average
        total = np.matrix(np.zeros((3, 1)), dtype=np.float)
        for X in entry["3Dlocs"]:
            total += X

        mean = total / len(entry["3Dlocs"])

        # update graph entry
        entry["3Dlocs"] = mean

        # determine color
        color = np.zeros((1, 3), dtype=np.float)
        for frame, pt2D in zip(entry["frames"], entry["2Dlocs"]):
            color += frames["images"][frame][int(pt2D[1, 0]), 
                                             int(pt2D[0, 0])].astype(np.float)

        color /= len(frames["images"])
        entry["color"] = color.astype(np.uint8)

def repeatedBundleAdjustment(graph, K, niter, freq, sd, 
                             percent_outliers, outlier_max_dist, max_err):
    """ Perform repeated bundle adjustment. """

    cnt = 0
    while True:
        cnt += 1
        print "\nBundle adjustment, ROUND %d." % cnt

        # every few rounds, remove outliers and jitter the initialization
        if cnt % freq == 0:
            outlierRejection(graph, K, percent_outliers, float("inf"))
            rms_error = bundleAdjustment(graph, K, niter, sd)
        else:
            rms_error = bundleAdjustment(graph, K, niter)
        
        if rms_error < max_err:
            break

#    outlierRejection(graph, K, 0.0, outlier_max_dist)



def bundleAdjustment(graph, K, niter=0, sd=0):
    """ Run bundle adjustment to joinly optimize camera poses and 3D points. """

    # unpack graph parameters into 1D array for initial guess
    x0, baseRt, keys, views, pts2D, pts3D = unpackGraph(graph)
    num_frames = len(graph["motion"])
    num_pts3D = len(pts3D)
    frameOffset = graph["frameOffset"]

    view_matrices, pts2D_matrices = createViewPointMatrices(views, pts2D, num_frames, 
                                                            num_pts3D, frameOffset)

    # insert Gaussian white noise
    noise = np.random.randn(len(x0)) * sd
    x0 += noise

    # run Levenberg-Marquardt algorithm
    global NUM_EVALS
    NUM_EVALS = 0

    args = (K, baseRt, view_matrices, pts2D_matrices, num_frames)
    result, success = leastsq(reprojectionError, x0, args=args, maxfev=niter)

    # get optimized motion and structure as lists
    optimized_motion = extractMotion(result, np.matrix(np.eye(3)),
                                     baseRt, num_frames)
    optimized_structure = np.hsplit(extractStructure(result, num_frames), num_pts3D)

    # update/repack graph
    graph["motion"] = optimized_motion
    for key, pt3D in zip(keys, optimized_structure):
        graph["3Dmatches"][key]["3Dlocs"] = pt3D

    return LAST_RMS_ERROR

def outlierRejection(graph, K, percent=5.0, max_dist=5.0):
    """ 
    Examine graph and remove some top percentage of outliers 
    and those outside a certain radius. 
    """

    # iterate through all points
    pq = PriorityQueue()
    marked_keys = []
    for key, entry in graph["3Dmatches"].iteritems():

        X = entry["3Dlocs"]

        # mark and continue if too far away from the origin
        if np.linalg.norm(X) > max_dist:
            marked_keys.append(key)
            continue

        # project into each frame
        errors = []
        for frame, x in zip(entry["frames"], entry["2Dlocs"]):
            frame -= graph["frameOffset"]
            Rt = graph["motion"][frame]

            proj = fromHomogenous(K * Rt * toHomogenous(X))
            diff = proj - x

            err = np.sqrt(np.multiply(diff, diff).sum())
            #print (frame, err)

            errors.append(err)

        # get mean error and add to priority queue
        # (priority is reciprocal of error since this is a MinPQ)
        mean_error = np.array(errors).mean()
        pq.put_nowait((1.0 / mean_error, key))

    # remove worst keys
    N = max(0, int((percent/100.0) * len(graph["3Dmatches"].keys())) - len(marked_keys))
    for i in range(N):
        score, key = pq.get_nowait()
        del graph["3Dmatches"][key]
        pq.task_done()

    # remove keys out of range
    for key in marked_keys:
        del graph["3Dmatches"][key]

    print "Removed %d outliers." % (N + len(marked_keys))

def unpackGraph(graph):
    """ Extract parameters for optimization. """

    # extract motion parameters
    baseRt = graph["motion"][0]

    poses = graph["motion"][1:]
    motion = []
    for p in poses:
        R = p[:, :-1]
        t = p[:, -1:]

        # convert to axis-angle format and concatenate
        r = toAxisAngle(R)
        motion.append(r)
        motion.append(np.array(t.T)[0, :])

    motion = np.hstack(motion)

    # extract frame parameters as array for all frames at each point
    keys = []
    views = []
    pts3D = []
    pts2D = []

    for key, entry in graph["3Dmatches"].iteritems():
        keys.append(key)
        views.append(entry["frames"])
        pts3D.append(entry["3Dlocs"])
        pts2D.append(entry["2Dlocs"])

    structure = np.array(pts3D).ravel()

    # concatenate motion/structure arrays
    x0 = np.hstack([motion, structure])

    return x0, baseRt, keys, views, pts2D, pts3D

def createViewPointMatrices(views, pts2D, num_frames, num_pts3D, frameOffset):
    """ Create frame-ordered lists of view and 2D point matrices. """

    # create lists of unpopulated matrices 
    view_matrices = []
    pts2D_matrices = []

    for i in range(num_frames):
        view_matrices.append(np.zeros(num_pts3D, dtype=np.bool))
        pts2D_matrices.append(np.matrix(np.zeros((2, num_pts3D)), dtype=np.float))

    # iterate through all 3D points and fill in matrices
    for i, (frames, pts) in enumerate(zip(views, pts2D)):
        for frame, pt in zip(frames, pts):
            frame -= frameOffset

            view_matrix = view_matrices[frame]
            pts2D_matrix = pts2D_matrices[frame]

            view_matrix[i] = True
            pts2D_matrix[:, i] = pt
            
    return view_matrices, pts2D_matrices

def reprojectionError(x, K, baseRt, view_matrices, pts2D_matrices, num_frames):
    """ Compute reprojection error for the graph with these parameters. """

    # unpack parameter vector
    motion_matrices = extractMotion(x, K, baseRt, num_frames)
    structure_matrix = toHomogenous(extractStructure(x, num_frames))

    # project all 3D points and store residuals according to view matrices
    residuals = []
    for P, views, pts2D in zip(motion_matrices, view_matrices, pts2D_matrices):
        proj = fromHomogenous(P * structure_matrix)

        # compute error
        diff = proj - pts2D

        # concatenate only the appropriate columns to residuals
        residuals.append(diff[:, views])

    # hstack, transpose, and ravel residuals
    error = np.asarray(np.hstack(residuals).T).ravel()
    rms_error = np.sqrt(np.multiply(error, error).sum()/(0.5*len(error)))
 
    global NUM_EVALS
    global LAST_RMS_ERROR
    if NUM_EVALS % 1000 == 0:
        print "Iteration %d, RMS error: %f" % (NUM_EVALS, rms_error)

    NUM_EVALS += 1
    LAST_RMS_ERROR = rms_error

    return error

def sigmoid(x):
    """ Sigmoid function to map angles to positive real numbers < 1."""

    return 1.0 / (1.0 + np.exp(x))

def inverseSigmoid(x):
    """ Invert sigmoid function. """

    return np.log((1.0 / x) - 1.0)

def toAxisAngle(R):
    """ 
    Decompose rotation R to axis-angle representation, where sigmoid(angle),
    is given as the magnitude of the axis vector.
    """

    # from https://en.wikipedia.org/wiki/Axis-angle-representation
    angle = np.arccos(0.5 * (np.trace(R) - 1.0))
    axis = 0.5/np.sin(angle) * np.array([R[2, 1] - R[1, 2],
                                         R[0, 2] - R[2, 0],
                                         R[1, 0] - R[0, 1]])

    return axis * sigmoid(angle)


def fromAxisAngle(r):
    """ Convert axis-angle representation to full rotation matrix. """

    # from https://en.wikipedia.org/wiki/Rotation_matrix
    angle = inverseSigmoid(np.linalg.norm(r))
    axis = r / np.linalg.norm(r)

    cross = np.matrix([[0, -axis[2], axis[1]],
                       [axis[2], 0, -axis[0]],
                       [-axis[1], axis[0], 0]], dtype=np.float)
    tensor = np.matrix([[axis[0]**2, axis[0]*axis[1], axis[0]*axis[2]],
                        [axis[0]*axis[1], axis[1]**2, axis[1]*axis[2]],
                        [axis[0]*axis[2], axis[1]*axis[2], axis[2]**2]], dtype=np.float)

    R = np.cos(angle)*np.eye(3) + np.sin(angle)*cross + (1-np.cos(angle))*tensor
 
    return R

def extractStructure(x, num_frames):
    """ Extract 3D points (as a single large matrix) from parameter vector. """

    # only consider the entries of x representing 3D structure
    offset = (num_frames - 1) * 6
    structure = x[offset:]

    # repack structure into 3D points
    num_pts = len(structure) / 3
    pts3D_matrix = np.matrix(structure.reshape(-1, 3)).T
 
    return pts3D_matrix

def extractMotion(x, K, baseRt, num_frames):
    """ 
    Extract camera poses (as a list of matrices) from parameter vector, 
    including implicit base pose. 
    """

    # only consider the entries of x representing poses
    offset = (num_frames - 1) * 6
    motion = x[:offset]

    # repack motion into 3x4 pose matrices
    pose_arrays = np.split(motion, num_frames - 1)
    pose_matrices = [K * baseRt]
    for p in pose_arrays:

        # convert from axis-angle to full pose matrix
        r = p[:3]
        t = p[3:]

        R = fromAxisAngle(r)
        pose_matrices.append(K * np.hstack([R, np.matrix(t).T]))

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

def inFront(Rt, X):
    """ Return true if X is in front of the camera. """

    R = Rt[:, :-1]
    t = Rt[:, -1:]

    if R[2, :] * (X + R.T * t) > 0:
        return True
    return False 

def triangulateLM(Rt1, Rt2, x1, x2, K):
    """ 
    Use nonlinear optimization to triangulate a 3D point, initialized with
    the estimate from triangulateCross().
    """

    # initialize with triangulateCross() linear solution
    x0 = np.asarray(triangulateCross(Rt1, Rt2, x1, x2, K).T)[0, :]

    # run Levenberg-Marquardt algorithm
    args = (Rt1, Rt2, x1, x2, K)
    result, success = leastsq(triangulationError, x0, args=args, maxfev=10000)

    # convert back to column vector and return
    X =  np.matrix(result).T

    # test reprojection
    #px1 = fromHomogenous(K * Rt1 * toHomogenous(X))
    #px2 = fromHomogenous(K * Rt2 * toHomogenous(X))

    #diff1 = px1 - x1
    #diff2 = px2 - x2
    #print "Errors (x1, x2): (%f, %f)" % (np.sqrt(np.multiply(diff1, diff1).sum()),
    #                                     np.sqrt(np.multiply(diff2, diff2).sum())) 

    return X

def triangulationError(x, Rt1, Rt2, x1, x2, K):
    """ Calculate triangulation error for single point x as a row-vector. """

    X = np.matrix(x).T

    # project into each frame
    px1 = fromHomogenous(K * Rt1 * toHomogenous(X))
    px2 = fromHomogenous(K * Rt2 * toHomogenous(X))

    # compute the diffs
    diff1 = px1 - x1
    diff2 = px2 - x2

    return np.asarray(np.vstack([diff1, diff2]).T)[0, :]

def triangulateLS(Rt1, Rt2, x1, x2, K):
    """ 
    Triangulate a least squares 3D point given two camera matrices
    and the point correspondence in non-homogenous coordinates.

    NOTE: This does not work very well due to ambiguity in homogenous coordinates.
    """
    
    A = np.vstack([K * Rt1, K * Rt2])
    b = np.vstack([toHomogenous(x1), toHomogenous(x2)])
    X = np.linalg.lstsq(A, b)[0]

    # testing
    px1 = fromHomogenous(K * Rt1 * toHomogenous(fromHomogenous(X)))
    px2 = fromHomogenous(K * Rt2 * toHomogenous(fromHomogenous(X)))

    diff1 = px1 - x1
    diff2 = px2 - x2
    print "Errors (x1, x2): (%f, %f)" % (np.sqrt(np.multiply(diff1, diff1).sum()),
                                         np.sqrt(np.multiply(diff2, diff2).sum())) 

    return X

def triangulateCross(Rt1, Rt2, x1, x2, K):
    """
    Triangulate a 3D point given its location in two frames of reference
    by using a cross product relation. Use least squares to solve.
    """

    # set up cross product matrix
    p1x = vector2cross(toHomogenous(x1))
    p2x = vector2cross(toHomogenous(x2))
    M = np.vstack([p1x * K * Rt1, p2x * K * Rt2])

    # solve with least squares
    A = M[:, :-1]
    b = -M[:, -1:]
    X = np.linalg.lstsq(A, b)[0]

    # testing
    #px1 = fromHomogenous(K * Rt1 * toHomogenous(X))
    #px2 = fromHomogenous(K * Rt2 * toHomogenous(X))

    #diff1 = px1 - x1
    #diff2 = px2 - x2
    #print "Errors (x1, x2): (%f, %f)" % (np.sqrt(np.multiply(diff1, diff1).sum()),
    #                                     np.sqrt(np.multiply(diff2, diff2).sum())) 

    return X

def vector2cross(v):
    """ Return the cross-product matrix version of a column vector. """

    cross = np.matrix([[0, -v[2, 0], v[1, 0]],
                       [v[2, 0], 0, -v[0, 0]],
                       [-v[1, 0], v[0, 0], 0]], dtype=np.float)
    return cross

def fromHomogenous(X):
    """ Transform a point from homogenous to normal coordinates. """

    x = X[:-1, :]
    x /= X[-1, :]

    return x

def toHomogenous(x):
    """ Transform a point from normal to homogenous coordinates. """

    X = np.vstack([x, np.ones((1, x.shape[1]))])

    return X

def basePose():
    """ Return the base camera pose. """

    return np.matrix(np.hstack([np.eye(3), np.zeros((3, 1))]))

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

def plotTrajectory(graph, max_dist=1e10):
    """ Show estimated camera trajectory. """

    tx = []
    ty = []
    tz = []
    for Rt in graph["motion"]:
        t = Rt[:, -1:]
        if np.linalg.norm(t) < max_dist:
            tx.append(t[0, 0])
            ty.append(t[1, 0])
            tz.append(t[2, 0])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(tx, ty, tz, marker="o")
    plt.show()

def showPointCloud(graph, max_dist=1e10):
    """ Show point cloud. """

    num_pts = len(graph["3Dmatches"].keys())

    px = np.zeros(num_pts, dtype=np.float)
    py = np.zeros(num_pts, dtype=np.float)
    pz = np.zeros(num_pts, dtype=np.float)
    colors = np.zeros((num_pts, 3), dtype=np.uint8)

    for i, (key, entry) in enumerate(graph["3Dmatches"].iteritems()):
        pt = entry["3Dlocs"]
        if np.linalg.norm(pt) < max_dist:
            px[i] = pt[0, 0]
            py[i] = pt[1, 0]
            pz[i] = pt[2, 0]
            #colors[i, :] = entry["color"].astype(np.float)/255.0
            #colors[i, :] = colors[i, [2, 1, 0]]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(px, py, pz, marker="o")
    plt.show()

def toPLY(graph, plyfile):
    """ Output graph structure to *.ply format. """

    num_pts = len(graph["3Dmatches"].keys())

    # create pts3D and color arrays
    pts3D_matrix = np.matrix(np.zeros((3, num_pts), dtype=np.float32))
    color_matrix = np.matrix(np.zeros((3, num_pts), dtype=np.uint8))

    for i, (key, entry) in enumerate(graph["3Dmatches"].iteritems()):
        pts3D_matrix[:, i] = entry["3Dlocs"]
        color_matrix[:, i] = entry["color"].T

    pts3D_matrix = pts3D_matrix.astype(np.float32)
    color_matrix = color_matrix.astype(np.uint8)

    # output to file
    f = open(plyfile, "wb")
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write("element vertex %d\n" % num_pts)
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar blue\n")
    f.write("property uchar green\n")
    f.write("property uchar red\n")
    f.write("end_header\n")
    for i in range(num_pts):
        pt = (pts3D_matrix[0, i], pts3D_matrix[1, i], pts3D_matrix[2, i], 
              color_matrix[0, i], color_matrix[1, i], color_matrix[2, i])
        f.write("%f %f %f %d %d %d\n" % pt)
    f.close()

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