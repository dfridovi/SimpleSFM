/*
 * Contains the definition for a Pose class to represent simple rotations and translations 
 * and perform elementary operations on both other poses and 3D points.
 */

#include "Pose.hpp"
#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;
using namespace std;

/*
private:
  Matrix4d Rt; // 4x4 homogeneous Pose matrix
  Vector6d aa; // axis-angle representation
*/

// Construct a new Pose from a rotation matrix and translation vector.
Pose::Pose(Matrix3d R, Vector3d t) {
  Rt = Matrix4d::Identity();
  Rt.block(0, 0, 3, 3) = R;
  Rt.col(3).head(3) = t;
  
  aa = new double[6];
}

// Destroy this Pose.
Pose::~Pose() {
  cout << "Deleting Pose..." << endl;

  delete[] aa;
}

// Project a 3D point into this Pose.
Vector2d Pose::project(Vector3d pt3d) {
  Vector4d pt3d_h = Vector4d::Constant(1.0);
  pt3d_h.head(3) = pt3d;
  
  Vector4d proj_h = Rt * pt3d_h;
  Vector2d proj = proj_h.head(2);
  proj /= proj_h(2);

  return proj;
}

// Compose this Pose with the given pose so that both refer to the identity Pose as 
// specified by the given Pose.
void Pose::compose(Pose p) {
  Rt *= p.Rt;
}

// Convert to axis-angle representation.
double* Pose::toAxisAngle() {
  return aa;
}

// Convert to matrix representation.
Matrix4d Pose::fromAxisAngle() {
  return Rt;
}

// Print to StdOut.
ostream& Pose::print(ostream& o)  {
  o << "Pose matrix:\n" << Rt << endl;
  o << "Pose axis-angle:\n";

  for (int i = 0; i < 6; i++)
    o << aa[i] << " ";

  o << endl;

  return o;
}

