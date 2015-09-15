/*
 * Header file for Graph.cpp, which contains a definition of the Graph class. The Graph class is
 * a compound object intended to track keypoint correspondences between two or more frames.
 */

#ifndef GRAPH
#define GRAPH

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <unordered_map>

#include "Pose.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;

class Graph {

public:

  // globally visible parameter vector
  vector<double> params; 

private:

  // store which frames a correspondence matches to
  unordered_map<Eigen::Vector2i, vector<int>> match_frames; 

  // store which 2D points a correspondence includes
  unordered_map<Eigen::Vector2i, vector<Eigen::Vector2i>> match_2Dlocs;

  // store the corresponding 3D location estimates
  unordered_map<Eigen::Vector2i, vector<Eigen::Vector3d>> match_3Dlocs;

  // store the next key for Graph merges
  unordered_map<Eigen::Vector2i, vector<Eigen::Vector2i>> match_nextKey;

  // keep track of Poses and initial frame offset for merging
  vector<Pose> motion;
  int frame_offset;

public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Create a new Graph "pair" from two Poses, a frame offset, and 
  // corresponding sets of 2D and 3D points.
  Graph(Pose &, Pose &, int, 
	vector<Eigen::Vector2i> &, vector<Eigen::Vector2i> &, vector<Eigen::Vector3d> &);

  // Destroy this Graph.
  ~Graph();

  // Combine a Graph with a "pair" (Graph with only two Poses).
  void updateGraph(Graph &);

  // Calculate reprojection error for this Graph based on its parameter vector.
  double reprojectionError();

  // Compute the parameter vector and update.
  void toParameterVector();

  // Convert parameter vector to full-blown Graph.
  void fromParameterVector();

  // Print out summary statistics.
  void printStatistics();
}

#endif
