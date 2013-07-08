#ifndef WATERSHED_H
#define WATERSHED_H

#include <CL/cl.hpp>
#include "utils/ocl_program_cache.h"

//void watershed(cl::CommandQueue& queue, cl::Kernel& kernel,
//            int width, int height,
//            cl::Buffer& src,// int src_pitch,
//            cl::Buffer& labeled);

void watershed(int width, int height,
               cl::Buffer& src,
               cl::Buffer& labeled,
               ProgramCache& cache = ProgramCache::getGlobalInstance(),
               cl::CommandQueue queue = ProgramCache::getGlobalInstance().getDefaultCommandQueue());


#endif // WATERSHED_H
