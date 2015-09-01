/*
 * Header file for BasicFunctions.cpp, which contains all of the major library functions
 * for the SimpleSFM project.
 */

#ifndef BASIC_FUNCTIONS
#define BASIC_FUNCTIONS

#include <Eigen/Dense>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "Graph.hpp"
#include "Pose.hpp"

using namespace Eigen;
using namespace Graph;
using namespace Pose;
using namespace std;
using namespace cv;

// Put all of this in its own namespace for easy access.
namespace bf {

public:

  // Convert focal length to camera intrinsic matrix.
  Matrix3d f2K(double);

  // Convert essential matrix to pair graph.
  Graph E2Rt(Matrix3d, Matrix3d, Pose, int, Vector2d, Vector2d, vector<DMatch>);

  // Run bundle adjustment to jointly optimize camera poses and 3D points.
  double bundleAdjustment(Graph, Matrix3d, int, double);

  // Compute reprojection error for the Graph with these parameters.
  // ************* FILL THIS IN ****************

private:

  // Use nonlinear optimization to triangulate a 3D point, intialized with a linear solution.
  Vector3d triangulateLM(Pose, Pose, Vector2d, Vector2d, Matrix3d);

  // Calculate triangulation error
  // ************* FILL THIS IN ****************

  // Triangulate a 3D point given its location in two frames of reference, using a cross product
  // relation and linear least squares.
  Vector3d triangulateCross(Pose, Pose, Vector2d, Vector2d, Matrix3d);

  // Return the cross-product matrix version of a 2D column vector (extending it to 
  // homogeneous coordinates).
  Matrix3d vector2cross(Vector2d);

}

#endif
