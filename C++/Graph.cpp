/*
 * Contains a definition of the Graph class. The Graph class is a compound object intended 
 * to track keypoint correspondences between two or more frames.
 */

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <unordered_map>
#include "Pose.hpp"

using namespace Eigen;
using namespace std;

// Create a new Graph from two Poses, a frame offset, and an OpenCV keypoint match vector.
Graph::Graph(Pose &p1, Pose &p2, int i, vector<DMatch> &matches) {
  motion.push_back(p1);
  potion.push_back(p2);
  frame_offset = i;

  // ----- FILL THIS IN ---------
}

// Destroy this Graph.
Graph::~Graph();

// Combine a Graph with a "pair" (Graph with only two Poses).
void Graph::updateGraph(Graph &);

// Calculate reprojection error for this Graph based on its parameter vector.
double Graph::reprojectionError();

// Compute the parameter vector and update.
void Graph::toParameterVector();

// Convert parameter vector to full-blown Graph.
void Graph::fromParameterVector();

// Print out summary statistics.
void Graph::printStatistics();

