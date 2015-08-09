"""
SimpleSFM main file.
"""

import os, sys
import BasicFunctions as bf
import numpy as np
import matplotlib.image 
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cPickle as pickle

# parameters
visualize = False
RATIO = 1.0
MIN_MATCHES = 10
PKLFILE = "pts3D.pkl"
PLYFILE = "model.ply"
LO_ITER = 20000
MAX_RMS_ERROR = 0.5
OUTLIER_MAX_DIST = 10
PERCENT_OUTLIERS = 10.0
NOISE_SD = 0.01
ADJUST_FREQ = 1

# set up
IMPATH = "Images/TestSeriesWatch/"
files = [f for f in os.listdir(IMPATH) if (not f.startswith(".") and not f == "model.ply")]

frames = {}
frames["files"] = np.array(files)
frames["images"] = []
frames["focal_length"] = RATIO * 4100.0 / 1.4
frames["K"] = bf.f2K(frames["focal_length"])
frames["num_images"] = len(files)

# graph dictionary
# -- motion is the frame-to-frame estimate of [R|t]
# -- 3Dmatches maps 3D points to 2D points in specific frames via a dictionary
#    whose keys are the tuple (kp_idx, most_recent_frame) and values are
#    the dict ([previous_frames], [xy_positions], [3D_estimates])
graph = {}
graph["motion"] = [bf.basePose()]
graph["3Dmatches"] = {}
graph["frameOffset"] = 0

# make an ORB detector
orb = cv2.ORB()

# make a brute force matcher
matcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING)

# keep track of last camera pose
lastRt = graph["motion"][0]

# for adjacent frames, detect ORB keypoints and extimate F
for i in range(1, frames["num_images"]):

    print "\nNow analyzing frames %d and %d of %d." % (i-1, i, frames["num_images"])

    # read in images
    img1 = cv2.imread(IMPATH + frames["files"][i - 1])
    img2 = cv2.imread(IMPATH + frames["files"][i])

    img1 = np.flipud(np.fliplr(cv2.resize(img1, dsize=(0, 0), fx=RATIO, fy=RATIO)))
    img2 = np.flipud(np.fliplr(cv2.resize(img2, dsize=(0, 0), fx=RATIO, fy=RATIO)))

    frames["images"].append(img1)
    if i == frames["num_images"] - 1:
        frames["images"].append(img2)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # get keypoints
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # do keypoint matching
    matches = matcher.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # estimate F if sufficient good matches
    if len(good_matches) < MIN_MATCHES:
        print "Did not find enough good matches (%d/%d)" % (len(good_matches),MIN_MATCHES)
        sys.exit(0)

    pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    inliers = [m for m, keep in zip(good_matches, matchesMask) if keep == 1]

    if visualize:
        bf.drawMatches(img1, kp1, img2, kp2, inliers)

    # get E from F and convert to pose, and get 3D pts as pair
    # a 'pair' is basically the same as a 'graph', but it has only two frames
    E = frames["K"].T * F * frames["K"]
    pair = bf.E2Rt(E, frames["K"], lastRt, i-1, kp1, kp2, inliers)

    print "Repeated pairwise bundle adjustment..."
    bf.repeatedBundleAdjustment(pair, frames["K"], LO_ITER, ADJUST_FREQ,
                                NOISE_SD, PERCENT_OUTLIERS, MAX_RMS_ERROR)
    lastRt = pair["motion"][1]

    # add pair
    bf.updateGraph(graph, pair)

# do bundle adjustment
bf.printGraphStats(graph)
bf.finalizeGraph(graph, frames)

print "Repeated global bundle adjustment..."
#bf.repeatedBundleAdjustment(graph, frames["K"], LO_ITER, ADJUST_FREQ,
#                            NOISE_SD, PERCENT_OUTLIERS, MAX_RMS_ERROR)

# pickle, just in case
f = open(PKLFILE, "wb")
pickle.dump(graph, f)
f.close()

# dense stereo matching

# output point cloud
bf.toPLY(graph, IMPATH + PLYFILE)

# plot camera translation over time
bf.plotTrajectory(graph)

# show point cloud
bf.showPointCloud(graph)