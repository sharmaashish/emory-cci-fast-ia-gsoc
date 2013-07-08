#ifndef PIXEL_OPS_H
#define PIXEL_OPS_H

#include <CL/cl.hpp>
#include "utils/ocl_program_cache.h"


void invert(int width, int height,
            cl::Buffer& src, int src_pitch,
            cl::Buffer& dst, int dst_pitch,
            ProgramCache& cache = ProgramCache::getGlobalInstance(),
            cl::CommandQueue queue = ProgramCache::getGlobalInstance().getDefaultCommandQueue());

void threshold(int width, int height,
               cl::Buffer& src, int src_pitch,
               cl::Buffer& dst, int dst_pitch,
               unsigned char lower, unsigned char upper,
               bool lower_inclusive, bool upper_inclusive,
               ProgramCache& cache = ProgramCache::getGlobalInstance(),
               cl::CommandQueue queue = ProgramCache::getGlobalInstance().getDefaultCommandQueue());

#endif //PIXEL_OPS_H
