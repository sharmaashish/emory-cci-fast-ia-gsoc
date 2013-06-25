#ifndef PIXEL_OPS_H
#define PIXEL_OPS_H

#include <CL/cl.hpp>

void invert(cl::CommandQueue& queue, cl::Kernel& kernel,
            int width, int height,
            cl::Buffer& src, int src_pitch,
            cl::Buffer& dst, int dst_pitch);

void threshold(cl::CommandQueue& queue, cl::Kernel& kernel,
            int width, int height,
            cl::Buffer& src, int src_pitch,
            cl::Buffer& dst, int dst_pitch,
            unsigned char lower, unsigned char upper,
            bool lower_inclusive, bool upper_inclusive);

#endif //PIXEL_OPS_H
