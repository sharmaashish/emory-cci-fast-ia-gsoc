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

/* Structure of 'data' buffer:
 *
 * |#######-------------|--------------------|
 *
 * # <- input data
 *
 * half of buffer is used as an input queue, the second part
 * as an output buffer, input data can be stored only in first half
 * of buffer (in input queue).
 */

void initQueueMetadata(int dataElements, int totalSize,
                       cl::Buffer& queueMetadata, cl::CommandQueue& queue)
{
    assert(!(totalSize & 1));
    assert(dataElements <= totalSize / 2);

    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

    int hostQueueMetadata[QUEUE_METATADATA_SIZE];

    assert(QUEUE_METATADATA_SIZE == 10);

    hostQueueMetadata[0] = dataElements; /* input data size */
    hostQueueMetadata[1] = 0;            /* input queue offset */
    hostQueueMetadata[2] = 0;            /* input queue head */
    hostQueueMetadata[3] = totalSize;    /* input/output queue max size */
    hostQueueMetadata[4] = 0;            /* output queue head */
    hostQueueMetadata[5] = totalSize/2;  /* output offset */
    hostQueueMetadata[6] = 0;            /* current input queue offset */
    hostQueueMetadata[7] = totalSize/2;  /* current output queue offset */
    hostQueueMetadata[8] = 0;            /* total inserts */
    hostQueueMetadata[9] = 0;            /* execution code */

    queueMetadata = cl::Buffer(context, CL_TRUE,
                      sizeof(hostQueueMetadata));

    queue.enqueueWriteBuffer(queueMetadata, CL_TRUE, 0, sizeof(hostQueueMetadata), hostQueueMetadata);

    //    std::stringstream params_stream;
    //    params_stream << "-DQUEUE_MAX_NUM_BLOCKS=" << QUEUE_MAX_NUM_BLOCKS << " ";
    //    params_stream << "-DQUEUE_NUM_THREADS=" << QUEUE_NUM_THREADS;

    //    std::string program_params = params_stream.str();

    ////    std::cout << "parallel queue ocl program params: " << program_params << std::endl;

    //    cl::Program& program = cache.getProgram("ParallelQueue", program_params);

    //   /* std::stringstream params_stream;
    //    params_stream << "-DQUEUE_MAX_NUM_BLOCKS=" << QUEUE_MAX_NUM_BLOCKS << " ";
    //    params_stream << "-DQUEUE_NUM_THREADS=" << QUEUE_NUM_THREADS;

    //    std::string program_params = params_stream.str();

    //    std::vector<std::string> sources;
    //    sources.push_back("ParallelQueue");
    //    sources.push_back("ParallelQueueTests");

    //    cl::Program& program = cache.getProgram(sources, program_params);
    //*/
    //    //////////////

    //    cl::Kernel init_queue_kernel(program, "init_queue_kernel");

    //    init_queue_kernel.setArg(0, queue_workspace);
    //    init_queue_kernel.setArg(1, inQueueData);
    //    init_queue_kernel.setArg(2, dataElements);
    //    init_queue_kernel.setArg(3, outQueueData);
    //    init_queue_kernel.setArg(4, outMaxSize);

    //    cl::NDRange global(1, 1);
    //    cl::NDRange local(1, 1);

    //    cl_int status = queue.enqueueNDRangeKernel(init_queue_kernel,
    //                                               cl::NullRange, global, local);
    //    assert(!status);
}



