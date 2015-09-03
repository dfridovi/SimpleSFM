/*
 * Header file for Graph.cpp, which contains a definition of the Graph class. The Graph class is
 * a compound object intended to track keypoint correspondences between two or more frames.
 */

#ifndef GRAPH
#define GRAPH

#include <Eigen/Dense>
#include <iostream>
#include "Pose.hpp"

using namespace Eigen;
using namespace std;
using namespace Pose;

class Graph {

private:
  unordered_map<Vector2i, list<int>> match_frames;
  unordered_map<Vector2i, list<Vector2i>> match_2Dlocs;
  unordered_map<Vector2i, list<Vector3d>> match_3Dlocs;
  list<Pose> motion;
  int frame_offset;

public:

  // Create a new Graph from two Poses and a frame offset..
  Graph(Pose, Pose, int);

  // Destroy this Graph.
  ~Graph();

  // Combine a Graph with a "pair" (Graph with only two Poses).
  void updateGraph(Graph);

  // Calculate reprojection error for this Graph.
  double reprojectionError();
}

#endif
