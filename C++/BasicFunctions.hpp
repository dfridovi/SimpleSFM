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
#include "ceres/ceres.h"

#include <vector>
#include <string>
#include <regex>

#include <boost/filesystem.hpp>

//#include "Graph.hpp"
#include "Pose.hpp"

using namespace std;
namespace fs = boost::filesystem;

namespace BasicFunctions {

  // Fetch all image file names.
  vector<string> getImgFiles(char *);
  
  // Convert focal length to camera intrinsic matrix.
  Eigen::Matrix3d f2K(double);

  // Convert fundamental matrix to essential matrix.
  Eigen::Matrix3d F2E(Eigen::Matrix3d &, Eigen::Matrix3d &);

  /*  
  // Convert essential matrix to pair graph.
  Graph::Graph E2Rt(Eigen::Matrix3d, Eigen::Matrix3d, Pose::Pose, int, 
  Eigen::Vector2i[], Eigen::Vector2i[], vector<cv::DMatch>);
  
  // Run bundle adjustment to jointly optimize camera poses and 3D points.
  double bundleAdjustment(Graph::Graph, Eigen::Matrix3d, int, double);
  
  // Use nonlinear optimization to triangulate a 3D point, intialized with a linear solution.
  Eigen::Vector3d triangulateLM(Pose::Pose, Pose::Pose, Eigen::Vector2i, 
  Eigen::Vector2d, Eigen::Matrix3d);
  
  // Calculate triangulation error
  double triangulationError(Pose::Pose, Eigen::Vector2i, Eigen::Vector2i, Eigen::Vector3d);
  
  // Triangulate a 3D point given its location in two frames of reference, using a cross product
  // relation and linear least squares.
  Eigen::Vector3d triangulateCross(Pose::Pose, Pose::Pose, Eigen::Vector2i, 
  Eigen::Vector2d, Eigen::Matrix3d);
  
  // Return the cross-product matrix version of a 2D column vector (extending it to 
  // homogeneous coordinates).
  Eigen::Matrix3d vector2cross(Eigen::Vector2d);
  */
}

#endif
