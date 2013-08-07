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

#define QUEUE_WARP_SIZE 	32
#define QUEUE_NUM_THREADS	512
#define QUEUE_NUM_WARPS (QUEUE_NUM_THREADS / QUEUE_WARP_SIZE)
#define LOG_QUEUE_NUM_THREADS 9
#define LOG_QUEUE_NUM_WARPS (LOG_QUEUE_NUM_THREADS - 5)
#define QUEUE_SCAN_STRIDE (QUEUE_WARP_SIZE + QUEUE_WARP_SIZE / 2 + 1)

void sumTest(cl::Buffer& device_result, int iterations,
                 ProgramCache& cache,
                 cl::CommandQueue& queue)
{
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

    std::stringstream params_stream;
    std::string program_params = params_stream.str();

    cl::Program& program = cache.getProgram("ParallelQueue", program_params);

    cl::Kernel sum_test_kernel(program, "sum_test");

    cl::Device device = queue.getInfo<CL_QUEUE_DEVICE>();

    int warp_size = sum_test_kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);

    std::cout << "warp size: " << warp_size << std::endl;

    cl::LocalSpaceArg local_queue = cl::__local(sizeof(int) * QUEUE_NUM_THREADS * 2);
    cl::LocalSpaceArg reduction_buffer = cl::__local(sizeof(int) * QUEUE_NUM_THREADS);
    cl::LocalSpaceArg got_work = cl::__local(sizeof(int));
    cl::LocalSpaceArg prefix_sum_input = cl::__local(sizeof(int) * QUEUE_NUM_THREADS);
    cl::LocalSpaceArg prefix_sum_output = cl::__local(sizeof(int) * QUEUE_NUM_THREADS);
    cl::LocalSpaceArg prefix_sum_workspace_1 = cl::__local(sizeof(int) * QUEUE_NUM_WARPS * QUEUE_SCAN_STRIDE);
    cl::LocalSpaceArg prefix_sum_workspace_2 = cl::__local(sizeof(int) * QUEUE_NUM_WARPS + QUEUE_NUM_WARPS / 2);

    sum_test_kernel.setArg(0, device_result);
    sum_test_kernel.setArg(1, iterations);
    sum_test_kernel.setArg(2, local_queue);
    sum_test_kernel.setArg(3, reduction_buffer);
    sum_test_kernel.setArg(4, got_work);
    sum_test_kernel.setArg(5, prefix_sum_input);
    sum_test_kernel.setArg(6, prefix_sum_output);
    sum_test_kernel.setArg(7, prefix_sum_workspace_1);
    sum_test_kernel.setArg(8, prefix_sum_workspace_2);

    cl::NDRange nullRange;
    cl::NDRange global(512, 1);
    cl::NDRange local(512, 1);

    cl_int status = queue.enqueueNDRangeKernel(sum_test_kernel,
                                               nullRange, global, local);
}

