"""
Test script.
"""

import numpy as np
import BasicFunctions as bf

f = 5.0
K = bf.f2K(f)
Rt1 = bf.basePose()

R2 = bf.fromAxisAngle(np.array([0, 0, np.pi/2.0]))
C2 = np.matrix([[f], [0], [0]])

Rt2 = np.hstack([R2, -R2 * C2])
x1 = np.matrix([[2.0*f],[0]])
x2 = np.matrix([[0],[f]])

print "x1 = " + str(x1.T)
print "x2 = " + str(x2.T)

p = bf.triangulateLM(Rt1, Rt2, x1, x2, K)

print "p = " + str(p.T)

px1 = bf.fromHomogenous(K * Rt1 * bf.toHomogenous(p))
px2 = bf.fromHomogenous(K * Rt2 * bf.toHomogenous(p))

print "px1 = " + str(px1.T)
print "px2 = " + str(px2.T)