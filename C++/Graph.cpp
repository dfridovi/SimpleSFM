/*
 * Contains a definition of the Graph class. The Graph class is a compound object intended 
 * to track keypoint correspondences between two or more frames.
 */

#include "Graph.hpp"

using namespace std;

// Create a new Graph "pair" from two Poses, a frame offset, and 
// corresponding sets of 2D and 3D points.
Graph::Graph(Pose &p1, Pose &p2, int frame, 
	     vector<Eigen::Vector2i> &pts2D_1, vector<Eigen::Vector2i> &pts2D_2, 
	     vector<Eigen::Vector3d> &pts3D) {

  // add poses and set frame offset
  motion.push_back(p1);
  potion.push_back(p2);
  frame_offset = frame;

  // populate hash tables indexed by the 2D point locations in the first frame
  for (int i = 0; i < pts3D.size(); i++) {

    // extract points
    Eigen::Vector2i pt1 = pts2D_1[i];
    Eigen::Vector2i pt2 = pts2D_2[i];
    Eigen::Vector3d pt3D = pts3D[i];

    // create hash entries
    vector<int> match_frames_entry;
    match_frames_entry.push_back(frame);
    match_frames_entry.push_back(frame + 1);

    vector<Eigen::Vector2i> match_2Dlocs_entry;
    match_frames_entry.push_back(pt1);
    match_frames_entry.push_back(pt2);

    vector<Eigen::Vector3d> match_3Dlocs_entry;
    match_frames_entry.push_back(pt3D);

    vector<Eigen::Vector2i> match_nextKey_entry;
    match_frames_entry.push_back(pt2);

  }
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

