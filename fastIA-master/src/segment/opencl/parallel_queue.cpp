#include "parallel_queue.h"
#include "utils/ocl_utils.h"

#include <iostream>


void initQueue(cl::Buffer& inQueueData, int dataElements,
               cl::Buffer& outQueueData, int outMaxSize,
               ProgramCache& cache,
               cl::CommandQueue& queue)
{
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

    std::stringstream params_stream;
    std::string program_params = params_stream.str();

    cl::Program& program = cache.getProgram("ParallelQueue", program_params);

    cl::Kernel init_queue_kernel(program, "init_queue_kernel");

    init_queue_kernel.setArg(0, inQueueData);
    init_queue_kernel.setArg(1, dataElements);
    init_queue_kernel.setArg(2, outQueueData);
    init_queue_kernel.setArg(3, outMaxSize);

    cl::NDRange nullRange;
    cl::NDRange global(1, 1);
    cl::NDRange local(1, 1);

    cl_int status = queue.enqueueNDRangeKernel(init_queue_kernel,
                                               nullRange, global, local);
}



void dequeueTest(cl::Buffer& device_result,
                 ProgramCache& cache,
                 cl::CommandQueue& queue)
{
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

    std::stringstream params_stream;
    std::string program_params = params_stream.str();

    cl::Program& program = cache.getProgram("ParallelQueue", program_params);

    cl::Kernel dequeue_test_kernel(program, "dequeue_test");

    cl::LocalSpaceArg local_mem = cl::__local(sizeof(int));

    dequeue_test_kernel.setArg(0, device_result);
    dequeue_test_kernel.setArg(1, local_mem);

    cl::NDRange nullRange;
    cl::NDRange global(512, 1);
    cl::NDRange local(512, 1);

    cl_int status = queue.enqueueNDRangeKernel(dequeue_test_kernel,
                                               nullRange, global, local);
}
