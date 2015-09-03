/*
 * Main file for SimpleSFM project. Run the entire SfM pipeline on a set of images and save the 
 * results to a *.ply file for viewing in MeshLab.
 */

#include <iostream>
#include <boost/filesystem.hpp>
#include "opencv2/features2d/features2d.hpp"
#include <vector>
#include <string>
#include <regex>

using namespace cv;
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
  vector<string> image_files;
  regex pattern("(.*)[(.jpg)|(.JPG)]");

  try {
    
    if (fs::exists(p) && fs::is_directory(p)) {
	
      // only add file paths if they match the regular expression
      for (fs::directory_entry &f : fs::directory_iterator(p)) {
	if (regex_match(f.path().string(), pattern))
	  image_files.push_back(f.path().string());
      }
      
    }
    
    else
      cout << p << " is not a valid directory.\n";
  }

  catch (const fs::filesystem_error &ex) {
      cout << ex.what() << '\n';
  }
      
  // print out the list of file names
  cout << "Image files:" << endl;
  for (int i = 0; i < image_files.size(); i++)
    cout << image_files[i] << endl;

  return 0;
}
