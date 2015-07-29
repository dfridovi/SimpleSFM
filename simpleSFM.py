"""
SimpleSFM main file. Based on SFMedu v2 by Jianxiong Xiao.
"""

import os, sys
import BasicFunctions as bf
import numpy as np
import matplotlib.image 
import cv2
import matplotlib.pyplot as plt

# parameters
visualize = False
RATIO = 0.2
MIN_MATCHES = 20

# set up
IMPATH = "Images/"
files = [f for f in os.listdir(IMPATH) if not f.startswith(".")]

frames = {}
frames["images"] = np.array(files)
frames["focal_length"] = 4100.0 / 1.4
frames["imsize"] = (2448, 3264)
frames["K"] = bf.f2K(frames["focal_length"])
frames["num_images"] = len(files)

# make an ORB detector
orb = cv2.ORB()

# make a brute force matcher
matcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING)

# for adjacent frames, detect ORB keypoints and extimate F
for i in range(frames["num_images"] - 1):

    # read in images
    img1 = cv2.imread(IMPATH + frames["images"][i])
    img2 = cv2.imread(IMPATH + frames["images"][i + 1])

    img1 = np.flipud(np.fliplr(cv2.resize(img1, dsize=(0, 0), fx=RATIO, fy=RATIO)))
    img2 = np.flipud(np.fliplr(cv2.resize(img2, dsize=(0, 0), fx=RATIO, fy=RATIO)))

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
    if len(good_matches) >= MIN_MATCHES:
        pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
    else:
        print "Did not find enough good matches (%d/%d)" % (len(good),MIN_MATCHES)
        sys.exit(0)
    
    if visualize:
        bf.drawMatches(img1, kp1, img2, kp2, good_matches, matchesMask)

    # get E from F and convert to poses
    