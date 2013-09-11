#ifndef COMPONENT_LABELING_H
#define COMPONENT_LABELING_H

#include <CL/cl.hpp>
#include "utils/ocl_program_cache.h"

void ccl(cl::Buffer src,
         cl::Buffer labels,
         int width, int height,
         int bgval,
         int connectivity,
         ProgramCache& cache = ProgramCache::getGlobalInstance(),
         cl::CommandQueue& queue = ProgramCache::getDefaultCommandQueue());

#endif // COMPONENT_LABELING_H
