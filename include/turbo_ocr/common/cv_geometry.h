// OpenCV geometry functions, whichever module they live in.
//
// OpenCV 5 moved the contour and 2-D transform helpers — arcLength,
// contourArea, minAreaRect, boundingRect, convexHull, approxPolyDP,
// getPerspectiveTransform, getRotationMatrix2D, moments and friends — out of
// imgproc into a new geometry module (opencv2/geometry/2d.hpp, linked as
// opencv_geometry). OpenCV 4 declares them in imgproc. Include this header
// instead of guessing; CMake links the module when the OpenCV in use has it.
#pragma once

#include <opencv2/imgproc.hpp>
#if __has_include(<opencv2/geometry/2d.hpp>)
#include <opencv2/geometry/2d.hpp>
#endif
