#ifndef PIXEL_OPS_H
#define PIXEL_OPS_H

#include <opencv2/ocl/ocl.hpp>
#include <opencv2/ocl/private/util.hpp>

using namespace cv;

void thresholdCaller(int rows, int cols, const ocl::oclMat img1,
 ocl::oclMat result, int lower, bool lower_inclusive, int upper, bool up_inclusive, cudaStream_t stream);


#endif //PIXEL_OPS_H
