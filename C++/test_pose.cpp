/*
 * Unit tests for Pose class.
 */

#include "Pose.hpp"
#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int main(void) {
  Matrix3d R = Matrix3d::Random();
  cout << "R:\n" << R << endl;

  Vector3d t = Vector3d::Random();
  cout << "t:\n" << t << endl;

  Pose p = Pose(R, t);
  p.print(cout);

  return 0;
}
