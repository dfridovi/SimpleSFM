/*
 * Contains the definition for a Pose class to represent simple rotations and translations 
 * and perform elementary operations on both other poses and 3D points.
 */

#include "Pose.hpp"

using namespace Eigen;
using namespace std;

// Construct a new Pose from a rotation matrix and translation vector.
Pose::Pose(Matrix3d &R, Vector3d &t) {
  Rt = Matrix4d::Identity();
  Rt.block(0, 0, 3, 3) = R;
  Rt.col(3).head(3) = t;
  
  aa = Vector3d::Zero();
}

// Destroy this Pose.
Pose::~Pose() {
  cout << "Deleting Pose..." << endl;
}

// Project a 3D point into this Pose.
Vector2d Pose::project(Vector3d &pt3d) {
  Vector4d pt3d_h = Vector4d::Constant(1.0);
  pt3d_h.head(3) = pt3d;
  
  Vector4d proj_h = Rt * pt3d_h;
  Vector2d proj = proj_h.head(2);
  proj /= proj_h(2);

  return proj;
}

// Compose this Pose with the given pose so that both refer to the identity Pose as 
// specified by the given Pose.
void Pose::compose(Pose &p) {
  Rt *= p.Rt;
}

// Convert to axis-angle representation.
VectorXd Pose::toAxisAngle() {
  
  // from https://en.wikipedia.org/wiki/Axis-angle-representation
  double angle = acos(0.5 * (Rt.trace() - 2.0));
  Vector3d axis = Vector3d(Rt(2, 1) - Rt(1, 2),
			   Rt(0, 2) - Rt(2, 0),
			   Rt(1, 0) - Rt(0, 1)) * 0.5 / sin(angle);

  axis /= axis.norm();
  aa = axis * angle;

  return aa;
}

// Convert to matrix representation.
Matrix4d Pose::fromAxisAngle() {

  // from https://en.wikipedia.org/wiki/Rotation_matrix
  double angle = aa.norm();
  Vector3d axis = aa / angle;

  Matrix3d cross;
  cross << 
    0.0, -axis(2), axis(1),
    axis(2), 0.0, -axis(0),
    -axis(1), axis(0), 0.0;

  Matrix3d tensor;
  tensor << 
    axis(0)*axis(0), axis(0)*axis(1), axis(0)*axis(2),
    axis(0)*axis(1), axis(1)*axis(1), axis(1)*axis(2),
    axis(0)*axis(2), axis(1)*axis(2), axis(2)*axis(2);

  Matrix3d R = cos(angle) * Matrix3d::Identity() + sin(angle) * cross + (1-cos(angle)) * tensor;
  Rt.block(0, 0, 3, 3) = R;

  return Rt;
}

// Print to StdOut.
void Pose::print()  {
  cout << "Pose matrix:\n" << Rt << endl;
  cout << "Pose axis-angle:\n" << aa << endl;
}

