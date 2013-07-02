#ifndef WATERSHED_H
#define WATERSHED_H

#include <CL/cl.hpp>


void watershed(cl::CommandQueue& queue, cl::Kernel& kernel,
            int width, int height,
            cl::Buffer& src,// int src_pitch,
            cl::Buffer& labeled);

#endif // WATERSHED_H
