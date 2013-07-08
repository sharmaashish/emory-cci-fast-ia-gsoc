#include "pixel-ops.h"
#include "utils/ocl_utils.h"

#include <iostream>

void invert(int width, int height,
            cl::Buffer& src, int src_pitch,
            cl::Buffer& dst, int dst_pitch,
            ProgramCache& cache,
            cl::CommandQueue queue)
{

    cl::Kernel kernel = cache.getKernel("Invert", "invert");

    kernel.setArg(0, src);
    kernel.setArg(1, src_pitch);
    kernel.setArg(2, dst);
    kernel.setArg(3, dst_pitch);
    kernel.setArg(4, width);
    kernel.setArg(5, height);

    cl::NDRange NullRange;
    cl::NDRange global(width, height);
    cl::NDRange local(1, 1);

    queue.enqueueNDRangeKernel(kernel, NullRange, global, local);
}


void threshold(int width, int height,
               cl::Buffer& src, int src_pitch,
               cl::Buffer& dst, int dst_pitch,
               unsigned char lower, unsigned char upper,
               bool lower_inclusive, bool upper_inclusive,
               ProgramCache& cache,
               cl::CommandQueue queue)
{

    cl::Kernel kernel = cache.getKernel("Threshold", "threshold");

    kernel.setArg(0, src);
    kernel.setArg(1, src_pitch);
    kernel.setArg(2, dst);
    kernel.setArg(3, dst_pitch);
    kernel.setArg(4, width);
    kernel.setArg(5, height);
    kernel.setArg(6, lower);
    kernel.setArg(7, upper);
    kernel.setArg(8, (unsigned char)lower_inclusive);
    kernel.setArg(9, (unsigned char)upper_inclusive);


    cl::NDRange NullRange;
    cl::NDRange global(width, height);
    cl::NDRange local(1, 1);

    queue.enqueueNDRangeKernel(kernel, NullRange, global, local);
}
