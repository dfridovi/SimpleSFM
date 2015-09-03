/*
 * Header file for Pose.cpp, which contains the definition for a Pose class to represent
 * simple rotations and translations and perform elementary operations on both other poses
 * and 3D points.
 */

#ifndef POSE
#define POSE

#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;
using namespace std;

class Pose {

private:
  Matrix4d Rt; // 4x4 homogeneous Pose matrix
  double* aa; // axis-angle representation

public:
  
  // Construct a new Pose from a rotation matrix and translation vector.
  Pose(Matrix3d, Vector3d);

  // Destroy this Pose.
  ~Pose();

  // Project a 3D point into this Pose.
  Vector2d project(Vector3d);

  // Compose this Pose with the given pose so that both refer to the identity Pose as 
  // specified by the given Pose.
  void compose(Pose);

  // Print to StdOut.
  ostream& print(ostream&);

private:

  // Convert to axis-angle representation.
  double* toAxisAngle();

  // Convert to matrix representation.
  Matrix4d fromAxisAngle();

};

#endif
