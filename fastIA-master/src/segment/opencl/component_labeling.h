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

int relabel(cl::Buffer labels,
            int width,
            int height,
            int bgval,
            ProgramCache& cache = ProgramCache::getGlobalInstance(),
            cl::CommandQueue& queue = ProgramCache::getDefaultCommandQueue());

void area_threshold(cl::Buffer labels, int width, int height,
                    int bgval, int min_size, int max_size,
                    ProgramCache& cache = ProgramCache::getGlobalInstance(),
                    cl::CommandQueue& queue = ProgramCache::getDefaultCommandQueue());


void bounding_box(cl::Buffer labels, int width, int height,
                  int bgval, int& count,
                  cl::Buffer out_labels,
                  cl::Buffer x_min, cl::Buffer x_max,
                  cl::Buffer y_min, cl::Buffer y_max,
                  ProgramCache& cache = ProgramCache::getGlobalInstance(),
                  cl::CommandQueue& queue = ProgramCache::getDefaultCommandQueue());



#endif // COMPONENT_LABELING_H
