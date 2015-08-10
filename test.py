"""
Test script.
"""

import numpy as np
import BasicFunctions as bf

f = 5.0
K = bf.f2K(f)
Rt1 = bf.basePose()

R21 = bf.fromAxisAngle(np.array([0, 0, np.pi/2.0]))
C21 = np.matrix([[f], [0], [0]])
Rt21 = np.hstack([R21, -R21 * C21])

"""
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
"""

R32 = np.matrix(np.eye(3))
C32 = np.matrix([[f], [0], [0]])
Rt32 = np.hstack([R32, -R32 * C32])

R31 = bf.fromAxisAngle(np.array([0, 0, np.pi/2.0]))
C31= np.matrix([[2*f], [0], [0]])
Rt31 = np.hstack([R31, -R31 * C31])

print Rt21 * np.vstack([Rt32, np.matrix([[0, 0, 0, 1]])])

Rt21inv = np.vstack([np.hstack([Rt21[:, :-1].T, 
                                 -Rt21[:, :-1].T * Rt21[:, -1:]]),
                     np.matrix([[0, 0, 0, 1]])])
Rt32inv = np.linalg.inv(np.vstack([Rt32, np.matrix([[0, 0, 0, 1]])]))

print np.linalg.inv(Rt32inv * Rt21inv)[:-1, :]

print Rt31