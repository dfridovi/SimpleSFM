/*
 * Unit tests for Pose class.
 */

#include "Pose.hpp"
#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int main(void) {
  Matrix3d R1;
  R1 << 
    cos(0.5), -sin(0.5), 0,
    sin(0.5), cos(0.5), 0,
    0, 0, 1;
  Vector3d t1 = Vector3d::Random();
  Pose p1 = Pose(R1, t1);
  p1.print();

  p1.toAxisAngle();
  p1.fromAxisAngle();
  p1.print();

  /*
  Matrix3d R2 = Matrix3d::Random();
  Vector3d t2 = Vector3d::Random();
  Pose p2 = Pose(R2, t2);
  p2.print();


  p2.toAxisAngle();
  p2.fromAxisAngle();

  p2.print();

  p2.compose(p1);
  p2.print();
  */
  return 0;
}
