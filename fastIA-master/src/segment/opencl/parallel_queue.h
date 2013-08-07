#ifndef PARALLEL_QUEUE_H
#define PARALLEL_QUEUE_H

#include <CL/cl.hpp>
#include "utils/ocl_program_cache.h"

void initQueueSystem(int queueMaxNumBlocks,
                     cl::CommandQueue& queue = ProgramCache::getGlobalInstance().getDefaultCommandQueue());

void disposeQueueSystem();

void initQueue(cl::Buffer& inQueueData, int dataElements,
               cl::Buffer& outQueueData, int outMaxSize,
               ProgramCache& cache = ProgramCache::getGlobalInstance(),
               cl::CommandQueue& queue = ProgramCache::getGlobalInstance().getDefaultCommandQueue());

void dequeueTest(cl::Buffer& device_result,
                 ProgramCache& cache = ProgramCache::getGlobalInstance(),
                 cl::CommandQueue& queue = ProgramCache::getGlobalInstance().getDefaultCommandQueue());

void sumTest(cl::Buffer& device_result, int iterations,
             ProgramCache& cache = ProgramCache::getGlobalInstance(),
             cl::CommandQueue& queue = ProgramCache::getGlobalInstance().getDefaultCommandQueue());

#endif // PARALLEL_QUEUE_H
