#ifndef PARALLEL_QUEUE_H
#define PARALLEL_QUEUE_H

#define QUEUE_MAX_NUM_BLOCKS	70
#define QUEUE_NUM_THREADS	    512

#define QUEUE_METATADATA_SIZE   10

#include <CL/cl.hpp>
#include "utils/ocl_program_cache.h"

//void initQueueSystem(cl::CommandQueue& queue = ProgramCache::getGlobalInstance()
//                        .getDefaultCommandQueue());

//void disposeQueueSystem();


void initQueueMetadata(int dataElements, int totalSize,
                cl::Buffer& queueMetadata,
                cl::CommandQueue& queue = ProgramCache::getGlobalInstance()
                                                    .getDefaultCommandQueue());

//void initQueue(const cl::Buffer& inQueueData, int dataElements,
//               const cl::Buffer& outQueueData, int outMaxSize,
//               ProgramCache& cache = ProgramCache::getGlobalInstance(),
//               cl::CommandQueue& queue = ProgramCache::getGlobalInstance()
//               .getDefaultCommandQueue());

#endif // PARALLEL_QUEUE_H
