#ifndef OCL_UTILS_H
#define OCL_UTILS_H

#include "ocl_program_cache.h"

#include <CL/cl.hpp>
#include <opencv/cv.hpp>

void oclSimpleInit(cl_device_type type, cl::Context& context, std::vector<cl::Device>& devices);
void oclPrintError(cl::Error& error);

cl::Buffer ocvMatToOclBuffer(cv::Mat& mat, cl::CommandQueue& queue);

void ocvMatToOclBuffer(cv::Mat& mat, cl::Buffer& buffer,
                       cl::Context& context, cl::CommandQueue& queue);

void oclBufferToOcvMat(cv::Mat& mat, cl::Buffer& buffer, int size, cl::CommandQueue& queue);
void oclBuferToOcvMat(cv::Mat& mat, cl::Buffer buffer, cl::CommandQueue queue);

#endif // OCL_UTILS_H
