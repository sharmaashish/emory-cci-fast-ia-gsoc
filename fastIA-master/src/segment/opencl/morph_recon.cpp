#include "morph_recon.h"
#include "parallel_queue.h"

#include <string>
#include <sstream>

#define DEBUG_PRINT

#define RECON_INIT_THREADS_X 16
#define RECON_INIT_THREADS_Y 16

extern cl::Buffer queue_workspace;


void morphReconInit(cl::Buffer marker, cl::Buffer mask,
                    int width, int height,
                    ProgramCache &cache, cl::CommandQueue &queue)
{
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

    std::stringstream params_stream;
    params_stream << "-DQUEUE_MAX_NUM_BLOCKS=" << QUEUE_MAX_NUM_BLOCKS;
    params_stream << " -DQUEUE_NUM_THREADS=" << QUEUE_NUM_THREADS;

    /* setting program parameters */
    std::string program_params = params_stream.str();

    std::cout << "parallel queue-based morphological reconstruction "
                 "ocl program params: " << program_params << std::endl;

    std::vector<std::string> sources;
    sources.push_back("ParallelQueue");
    sources.push_back("MorphRecon");

    cl::Program& program = cache.getProgram(sources, program_params);

    /* obtaining kernel */
    cl::Kernel scan_forward_rows_kernel(program, "scan_forward_rows_kernel");

    /* allocating buffers */

    cl::Buffer changed(context, CL_TRUE, sizeof(int));

    cl::LocalSpaceArg marker_local = cl::__local(sizeof(int)
                                                 * RECON_INIT_THREADS_X
                                                 * RECON_INIT_THREADS_Y);

    cl::LocalSpaceArg mask_local = cl::__local(sizeof(int)
                                                 * RECON_INIT_THREADS_X
                                                 * RECON_INIT_THREADS_Y);

    scan_forward_rows_kernel.setArg(0, marker);
    scan_forward_rows_kernel.setArg(1, mask);
    scan_forward_rows_kernel.setArg(2, changed);
    scan_forward_rows_kernel.setArg(3, marker_local);
    scan_forward_rows_kernel.setArg(4, mask_local);
    scan_forward_rows_kernel.setArg(5, width);
    scan_forward_rows_kernel.setArg(6, height);


    /* calculating kernel dimensions */

    cl::NDRange local(RECON_INIT_THREADS_X, RECON_INIT_THREADS_Y);

    int global_x = RECON_INIT_THREADS_X;
    int global_y = ((height + RECON_INIT_THREADS_Y - 1) / RECON_INIT_THREADS_Y)
                                                        * RECON_INIT_THREADS_Y;
    cl::NDRange global(global_x, global_y);

#ifdef DEBUG_PRINT
    std::cout << "global_x: " << global_x
              <<", global_y: " << global_y << std::endl;
#endif

    cl_int status = queue.enqueueNDRangeKernel(scan_forward_rows_kernel,
                                               cl::NullRange, global, local);
}


void morphRecon(cl::Buffer inputQueueData, int dataElements, int queueSize,
                cl::Buffer seeds, cl::Buffer image, int ncols, int nrows,
                ProgramCache &cache, cl::CommandQueue &queue)
{
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

//    initQueueSystem(queue); /* initializing queue */

//    int output_buffer_size = 512;
//    cl::Buffer output_buffer(context, CL_TRUE,
//                             sizeof(int) * output_buffer_size);

//    initQueue(input_list, dataElements, output_buffer, output_buffer_size,
//              cache, queue);

    cl::Buffer device_queue_metadata;

    // initialize queue metadata buffer
    initQueueMetadata(dataElements, queueSize, device_queue_metadata, queue);

    std::stringstream params_stream;
    params_stream << "-DQUEUE_MAX_NUM_BLOCKS=" << QUEUE_MAX_NUM_BLOCKS;
    params_stream << " -DQUEUE_NUM_THREADS=" << QUEUE_NUM_THREADS;
   // params_stream << " -Werror";
//    params_stream << " -cl-opt-disable";


    std::string program_params = params_stream.str();

 //   std::cout << "parallel queue-based morphological reconstruction "
   //              "ocl program params: " << program_params << std::endl;

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
    morph_recon_kernel.setArg(5, inputQueueData);
    morph_recon_kernel.setArg(6, device_queue_metadata);
    morph_recon_kernel.setArg(7, local_queue);
    morph_recon_kernel.setArg(8, reduction_buffer);
    morph_recon_kernel.setArg(9, got_work);
    morph_recon_kernel.setArg(10, prefix_sum_input);
    morph_recon_kernel.setArg(11, prefix_sum_output);

    cl::NDRange global(QUEUE_NUM_THREADS, 1);
    cl::NDRange local(QUEUE_NUM_THREADS, 1);

#ifdef DEBUG_PRINT
    std::cout << "running morphological reconstruction kernel..."
              << std::endl;
#endif

    cl_int status = queue.enqueueNDRangeKernel(morph_recon_kernel,
                                               cl::NullRange, global, local);

//    disposeQueueSystem();
}
