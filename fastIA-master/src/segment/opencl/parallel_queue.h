#ifndef PARALLEL_QUEUE_H
#define PARALLEL_QUEUE_H

#define QUEUE_MAX_NUM_BLOCKS	70
#define QUEUE_NUM_THREADS	    512

//#define QUEUE_METATADATA_SIZE   10

#include <CL/cl.hpp>
#include "utils/ocl_program_cache.h"

void initQueueMetadata(int dataElements, int totalSize,
              cl::Buffer& queueMetadata, cl::Buffer& executionCode,
              cl::CommandQueue& queue = ProgramCache::getDefaultCommandQueue());

void initQueueMetadata(std::vector<int>& dataElements,
              std::vector<int>& totalSizes,
              cl::Buffer& queueMetadata,
              cl::Buffer& executionCode,
              cl::CommandQueue& queue = ProgramCache::getDefaultCommandQueue());


#endif // PARALLEL_QUEUE_H
