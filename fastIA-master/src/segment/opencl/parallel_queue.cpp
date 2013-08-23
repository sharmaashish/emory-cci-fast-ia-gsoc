#include "parallel_queue.h"
#include "utils/ocl_utils.h"

#include <iostream>

cl::Buffer queue_workspace;

void initQueueSystem(cl::CommandQueue& queue)
{
    cl::Device device = queue.getInfo<CL_QUEUE_DEVICE>();
    int addressSize = device.getInfo<CL_DEVICE_ADDRESS_BITS>()/8;

    int byteSize = 5 * sizeof(int) * QUEUE_MAX_NUM_BLOCKS
            + 4 * addressSize * QUEUE_MAX_NUM_BLOCKS + sizeof(int);

    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
    queue_workspace = cl::Buffer(context, CL_TRUE, byteSize);
}

void disposeQueueSystem()
{
    queue_workspace = cl::Buffer();
}

void initQueue(const cl::Buffer& inQueueData, int dataElements,
               const cl::Buffer& outQueueData, int outMaxSize,
               ProgramCache& cache,
               cl::CommandQueue& queue)
{
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

    std::stringstream params_stream;
    params_stream << "-DQUEUE_MAX_NUM_BLOCKS=" << QUEUE_MAX_NUM_BLOCKS << " ";
    params_stream << "-DQUEUE_NUM_THREADS=" << QUEUE_NUM_THREADS;

    std::string program_params = params_stream.str();

//    std::cout << "parallel queue ocl program params: " << program_params << std::endl;

    cl::Program& program = cache.getProgram("ParallelQueue", program_params);

   /* std::stringstream params_stream;
    params_stream << "-DQUEUE_MAX_NUM_BLOCKS=" << QUEUE_MAX_NUM_BLOCKS << " ";
    params_stream << "-DQUEUE_NUM_THREADS=" << QUEUE_NUM_THREADS;

    std::string program_params = params_stream.str();

    std::vector<std::string> sources;
    sources.push_back("ParallelQueue");
    sources.push_back("ParallelQueueTests");

    cl::Program& program = cache.getProgram(sources, program_params);
*/
    //////////////

    cl::Kernel init_queue_kernel(program, "init_queue_kernel");

    init_queue_kernel.setArg(0, queue_workspace);
    init_queue_kernel.setArg(1, inQueueData);
    init_queue_kernel.setArg(2, dataElements);
    init_queue_kernel.setArg(3, outQueueData);
    init_queue_kernel.setArg(4, outMaxSize);

    cl::NDRange global(1, 1);
    cl::NDRange local(1, 1);

    cl_int status = queue.enqueueNDRangeKernel(init_queue_kernel,
                                               cl::NullRange, global, local);
    assert(!status);
}



