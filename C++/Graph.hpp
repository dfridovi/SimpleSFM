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

using namespace Eigen;
using namespace std;

class Graph {

public:
  vector<double> params; // globally visible parameter vector

private:
  unordered_map<Vector2i, vector<int>> match_frames;
  unordered_map<Vector2i, vector<Vector2i &>> match_2Dlocs;
  unordered_map<Vector2i, vector<Vector3d &>> match_3Dlocs;
  vector<Pose &> motion;
  int frame_offset;

public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Create a new Graph "pair" from two Poses, a frame offset, and an OpenCV keypoint match vector.
  Graph(Pose &, Pose &, int, vector<DMatch> &);

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
