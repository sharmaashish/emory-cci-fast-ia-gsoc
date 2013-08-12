#include "morph_recon.h"
#include "parallel_queue.h"

#include <string>
#include <sstream>

#define DEBUG_PRINT

extern cl::Buffer queue_workspace;

void morphRecon(cl::Buffer input_list, int dataElements, cl::Buffer seeds,
                cl::Buffer image, int ncols, int nrows,
                ProgramCache &cache, cl::CommandQueue &queue)
{
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

    initQueueSystem(queue); /* initializing queue */

    int output_buffer_size = 512;
    cl::Buffer output_buffer(context, CL_TRUE,
                             sizeof(int) * output_buffer_size);

    initQueue(input_list, dataElements, output_buffer, output_buffer_size,
              cache, queue);

    std::stringstream params_stream;
    params_stream << "-DQUEUE_MAX_NUM_BLOCKS=" << QUEUE_MAX_NUM_BLOCKS;
    params_stream << " -DQUEUE_NUM_THREADS=" << QUEUE_NUM_THREADS;

    std::string program_params = params_stream.str();

    std::cout << "parallel queue-based morphological reconstruction "
                 "ocl program params: " << program_params << std::endl;

    std::vector<std::string> sources;
    sources.push_back("ParallelQueue");
    sources.push_back("MorphRecon");

    cl::Program& program = cache.getProgram(sources, program_params);

    cl::Kernel morph_recon_kernel(program, "morph_recon_kernel");

    cl::Buffer device_total_elements(context, CL_TRUE, sizeof(int));

    cl::LocalSpaceArg local_queue
            = cl::__local(sizeof(int)* QUEUE_NUM_THREADS * 5);
    cl::LocalSpaceArg reduction_buffer
            = cl::__local(sizeof(int) * QUEUE_NUM_THREADS);
    cl::LocalSpaceArg got_work
            = cl::__local(sizeof(int));
    cl::LocalSpaceArg prefix_sum_input
            = cl::__local(sizeof(int) * QUEUE_NUM_THREADS);
    cl::LocalSpaceArg prefix_sum_output
            = cl::__local(sizeof(int) * QUEUE_NUM_THREADS);

    morph_recon_kernel.setArg(0, device_total_elements);
    morph_recon_kernel.setArg(1, seeds);
    morph_recon_kernel.setArg(2, image);
    morph_recon_kernel.setArg(3, ncols);
    morph_recon_kernel.setArg(4, nrows);
    morph_recon_kernel.setArg(5, queue_workspace);
    morph_recon_kernel.setArg(6, local_queue);
    morph_recon_kernel.setArg(7, reduction_buffer);
    morph_recon_kernel.setArg(8, got_work);
    morph_recon_kernel.setArg(9, prefix_sum_input);
    morph_recon_kernel.setArg(10, prefix_sum_output);

    cl::NDRange global(QUEUE_NUM_THREADS, 1);
    cl::NDRange local(QUEUE_NUM_THREADS, 1);

#ifdef DEBUG_PRINT
    std::cout << "running morphological reconstruction kernel..."
              << std::endl;
#endif

    cl_int status = queue.enqueueNDRangeKernel(morph_recon_kernel,
                                               cl::NullRange, global, local);

    disposeQueueSystem();
}
