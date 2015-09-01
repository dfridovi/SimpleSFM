# SimpleSFM
A simple structure from motion library.

## Current Status
Right now, the Python side of SimpleSFM is reasonably stable. Essentially, it can process a series of still images from an iPhone 5 and output an optimized 3D sparse point cloud. It's not particularly fast, but I've done my best to write good, clean, well-documented code. I should add that, in its current incarnation, I can imagine several things that might go wrong in certain cases which might effect the results. However, without better visualization (i.e. involving dense matching) it is very hard, if not impossible to know for certain.

## Future Goals
My vision for this project is that it should support dense matching between images and be reimplemented in C++ both for speed and to allow more concise expression of the bundle adjustment problem (i.e. calculation of reprojection error without explicit vectorization). To that end, here is my priority queue:

1. Reimplement the Python code in C++.
   a. Do a careful re-evaluation of appropriate data structures.
   b. Rewrite Python code accordingly, with more rigorous APIs.
4. Add dense matching.