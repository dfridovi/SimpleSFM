"""
SimpleSFM main file. Based on SFMedu v2 by Jianxiong Xiao.
"""

import os
import BasicFunctions as bf
import numpy as np
import matplotlib.image 
import cv2
import matplotlib.pyplot as plt

visualize = True

# set up
IMPATH = "../Images/"
files = os.listdir(IMPATH)

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
	img1 = cv2.cvtColor(cv2.imread(IMPATH + frames["images"][i]), cv2.COLOR_BGR2GRAY)
	img2 = cv2.cvtColor(cv2.imread(IMPATH + frames["images"][i+1]), cv2.COLOR_BGR2GRAY)

	# get keypoints
	kp1, des1 = orb.detectAndCompute(img1, None)
	kp2, des2 = orb.detectAndCompute(img2, None)

	# do keypoint matching
	matches = matcher.knnMatch(des1, des2, k=2)

	# store all the good matches as per Lowe's ratio test.
	good_matches = []
	for m, n in matches:
	    if m.distance < 0.7*n.distance:
        	good_matches.append([m])

	if visualize:
		bf.drawMatches(img1, kp1, img2, kp2, good_matches)