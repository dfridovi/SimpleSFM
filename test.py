"""
Test script.
"""

import numpy as np
import BasicFunctions as bf

f = 1.0
K = bf.f2K(f)
Rt1 = bf.basePose()
Rt2a = np.hstack([np.matrix(np.eye(3)), np.matrix([[-f], [0], [0]])])
Rt2b = np.hstack([np.matrix(np.eye(3)), np.matrix([[f], [0], [0]])])
x1 = np.matrix([[f],[0]])
x2 = np.matrix([[0],[0]])

print "x1 = " + str(x1.T)
print "x2 = " + str(x2.T)

p = bf.triangulateCross(Rt1, Rt2a, x1, x2, K)

print "p = " + str(p.T)

px1 = bf.fromHomogenous(K * Rt1 * bf.toHomogenous(p))
px2 = bf.fromHomogenous(K * Rt2a * bf.toHomogenous(p))

print "px1 = " + str(px1.T)
print "px2 = " + str(px2.T)

print bf.fromHomogenous(K * Rt1 * np.matrix([[f, 0, f, 1.0]]).T)
print bf.fromHomogenous(K * Rt2a * np.matrix([[f, 0, f, 1.0]]).T)