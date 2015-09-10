/*
 * Main file for SimpleSFM project. Run the entire SfM pipeline on a set of images and save the 
 * results to a *.ply file for viewing in MeshLab.
 */

#include <iostream>
#include <boost/filesystem.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include <string>
#include <regex>

using namespace std;
namespace fs = boost::filesystem;

void usage(char *argv[]) {
  cerr << "Usage: " << argv[0] << " PATH_TO_IMAGES" << endl;
}

int main(int argc, char *argv[]) {

  // ensure proper utilization
  if (argc != 2) {
    usage(argv);
    return 1;
  }

  // parse file names in supplied directory
  fs::path p(argv[1]);
  vector<string> img_files;
  regex pattern(".*\\.((JPG)|(jpg)|(jpeg))");

  try {
    
    if (fs::exists(p) && fs::is_directory(p)) {
	
      // only add file paths if they match the regular expression
      for (fs::directory_entry &f : fs::directory_iterator(p)) {
	if (regex_match(f.path().string(), pattern))
	  img_files.push_back(f.path().string());
      }
      
    }
    
    else
      cout << p << " is not a valid directory.\n";
  }

  catch (const fs::filesystem_error &ex) {
      cout << ex.what() << '\n';
  }
      
  // print out the list of file names
  const int num_files = img_files.size();
  cout << "Image files:" << endl;
  for (int i = 0; i < num_files; i++)
    cout << img_files[i] << endl;

  // iterate through each pair of consecutive images and find keypoints
  for (int i = 1; i < num_files; i++) {
    cout << "\nNow analyzing frames " << i-1 << " and " << i << " of " << num_files-1 << ".\n";
    
    cv::Mat img1 = cv::imread(img_files[i-1], CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat img2 = cv::imread(img_files[i], CV_LOAD_IMAGE_GRAYSCALE);

    if (!img1.data || !img2.data) {
      cerr << "Error reading images." << endl; 
      return -1; 
    }

    // detect and compute keypoints and descriptors
    cv::ORB orb;

    vector<cv::KeyPoint> kp1, kp2;
    cv::Mat des1, des2;

    orb.detect(img1, kp1); orb.compute(img1, kp1, des1);
    orb.detect(img2, kp2); orb.compute(img2, kp2, des2);

    // print out some summary information
    cout << "Found " << kp1.size() << " keypoints in image 1." << endl;
    cout << "des1 shape: (" << des1.rows << ", " << des1.cols << ")" << endl;

    cout << "Found " << kp2.size() << " keypoints in image 2." << endl;
    cout << "des2 shape: (" << des2.rows << ", " << des2.cols << ")" << endl;

    // do keypoint matching
    cv::BFMatcher matcher;
    vector<vector<cv::DMatch>> matches;
    matcher.knnMatch(des1, des2, matches, 2);
  }

  return 0;
}
