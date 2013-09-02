#ifndef MORPH_RECON_H
#define MORPH_RECON_H

#include <CL/cl.hpp>
#include "utils/ocl_program_cache.h"
#include "utils/ocl_type_resolver.h"

#include "parallel_queue.h"

#include <sstream>
#include <cassert>
#include <vector>

#define DEBUG_PRINT
#define WARNINGS_AS_ERRORS

#define INIT_SCAN_X_THREADS_X 16
#define INIT_SCAN_X_THREADS_Y 16

#define INIT_SCAN_Y_THREAD_X 256

#define QUEUE_EXPAND_FACTOR 2

static std::stringstream& initMorphReconProgramParams()
{
    static std::stringstream params_stream;
    static int initialized = false;

    if(!initialized)
    {
        params_stream << "-DQUEUE_MAX_NUM_BLOCKS=" << QUEUE_MAX_NUM_BLOCKS;
        params_stream << " -DQUEUE_NUM_THREADS=" << QUEUE_NUM_THREADS;

#ifdef WARNINGS_AS_ERRORS
        params_stream << " -Werror";
#endif
        initialized = true;
    }

    return params_stream;
}

static std::string morphReconParams = initMorphReconProgramParams().str();

template<typename MARKER_TYPE, typename MASK_TYPE>
static std::string morphReconTypeParams()
{
    std::string typeParam = " -DMARKER_TYPE="
            + TypeResolver<MARKER_TYPE>::type_as_string
            + " -DMASK_TYPE=" + TypeResolver<MASK_TYPE>::type_as_string;

    return typeParam;
}


/**
 * @brief First step in morphological reconstruction. There are
 *        performed forward and backward scans, separately for rows and colums.
 *        This functions is for internal usage only.
 *
 * @param marker    Buffer with marker
 * @param mask      Buffer with mask
 * @param width     Marker and mask width
 * @param height    Marker and mask height
 * @param queue     Queue used to execute kernels
 * @param program   Program from which kernels are obtained
 */
template <typename MARKER_TYPE, typename MASK_TYPE>
void morphReconInitScan(cl::Buffer marker, cl::Buffer mask,
                        int width, int height,
                        cl::CommandQueue &queue, cl::Program &program)
{
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

    /* dimensions for x scans */

    cl::NDRange local_scan_x(INIT_SCAN_X_THREADS_X, INIT_SCAN_X_THREADS_Y);

    int global_x_scan_x = INIT_SCAN_X_THREADS_X;
    int global_y_scan_x = ((height + INIT_SCAN_X_THREADS_Y - 1)
                          / INIT_SCAN_X_THREADS_Y) * INIT_SCAN_X_THREADS_Y;

    cl::NDRange global_scan_x(global_x_scan_x, global_y_scan_x);

#ifdef DEBUG_PRINT
    std::cout << "kernel dimensions for scan_x (forward and backward): "
              << "global_x: " << global_x_scan_x << ", "
              << "global_y: " << global_y_scan_x << std::endl;
#endif

    /* dimensions for y scans */

    cl::NDRange local_scan_y(INIT_SCAN_Y_THREAD_X, 1);

    int global_x_scan_y = ((width + INIT_SCAN_Y_THREAD_X - 1)
                           / INIT_SCAN_Y_THREAD_X) * INIT_SCAN_Y_THREAD_X;

    cl::NDRange global_scan_y(global_x_scan_y, 1);

#ifdef DEBUG_PRINT
    std::cout << "kernel dimensions for scan_y (forward and backward): "
              << "global_x: " << global_x_scan_y << ", "
              << "global_y: " << 1 << std::endl;
#endif

    /******************/
    /* SCAN FORWARD X */
    /******************/

    /* obtaining kernel */
    cl::Kernel scan_forward_rows_kernel(program, "scan_forward_rows_kernel");

    /* allocating buffers */

    cl::Buffer changed(context, CL_TRUE, sizeof(int));

    cl::LocalSpaceArg marker_local = cl::__local(sizeof(MARKER_TYPE)
                                                 * INIT_SCAN_X_THREADS_X
                                                 * INIT_SCAN_X_THREADS_Y);

    cl::LocalSpaceArg mask_local = cl::__local(sizeof(MASK_TYPE)
                                                 * INIT_SCAN_X_THREADS_X
                                                 * INIT_SCAN_X_THREADS_Y);

    scan_forward_rows_kernel.setArg(0, marker);
    scan_forward_rows_kernel.setArg(1, mask);
    scan_forward_rows_kernel.setArg(2, changed);
    scan_forward_rows_kernel.setArg(3, marker_local);
    scan_forward_rows_kernel.setArg(4, mask_local);
    scan_forward_rows_kernel.setArg(5, width);
    scan_forward_rows_kernel.setArg(6, height);

    queue.enqueueNDRangeKernel(scan_forward_rows_kernel,
                                            cl::NullRange,
                                            global_scan_x, local_scan_x);

    /******************/
    /* SCAN FORWARD Y */
    /******************/

    /* obtaining kernel */
    cl::Kernel scan_forward_columns_kernel(program,
                                           "scan_forward_columns_kernel");

    scan_forward_columns_kernel.setArg(0, marker);
    scan_forward_columns_kernel.setArg(1, mask);
    scan_forward_columns_kernel.setArg(2, changed);
    scan_forward_columns_kernel.setArg(3, width);
    scan_forward_columns_kernel.setArg(4, height);


    queue.enqueueNDRangeKernel(scan_forward_columns_kernel,
                               cl::NullRange, global_scan_y, local_scan_y);

    /*******************/
    /* SCAN BACKWARD X */
    /*******************/

    /* obtaining kernel */
    cl::Kernel scan_backward_rows_kernel(program, "scan_backward_rows_kernel");

    scan_backward_rows_kernel.setArg(0, marker);
    scan_backward_rows_kernel.setArg(1, mask);
    scan_backward_rows_kernel.setArg(2, changed);
    scan_backward_rows_kernel.setArg(3, marker_local);
    scan_backward_rows_kernel.setArg(4, mask_local);
    scan_backward_rows_kernel.setArg(5, width);
    scan_backward_rows_kernel.setArg(6, height);

    queue.enqueueNDRangeKernel(scan_backward_rows_kernel,
                               cl::NullRange, global_scan_x, local_scan_x);

    /*******************/
    /* SCAN BACKWARD Y */
    /*******************/

    /* obtaining kernel */
    cl::Kernel scan_backward_columns_kernel(program,
                                            "scan_backward_columns_kernel");

    scan_backward_columns_kernel.setArg(0, marker);
    scan_backward_columns_kernel.setArg(1, mask);
    scan_backward_columns_kernel.setArg(2, changed);
    scan_backward_columns_kernel.setArg(3, width);
    scan_backward_columns_kernel.setArg(4, height);

    queue.enqueueNDRangeKernel(scan_backward_columns_kernel,
                               cl::NullRange, global_scan_y, local_scan_y);
}

/**
 * @brief See: morphReconInitScan(cl::Buffer, cl::Buffer, int,
 *        int, cl::CommandQueue&, cl::Program&);
 *        This function is exposed for testing purposes
 */
template <typename MARKER_TYPE, typename MASK_TYPE>
void morphReconInitScan(cl::Buffer marker, cl::Buffer mask,
               int width, int height,
               ProgramCache& cache = ProgramCache::getGlobalInstance(),
               cl::CommandQueue& queue = ProgramCache::getDefaultCommandQueue())
{
    std::cout << "parallel queue-based morphological reconstruction, init scan "
                 "ocl program params: " << morphReconParams << std::endl;

    std::vector<std::string> sources;
    sources.push_back("ParallelQueue");
    sources.push_back("MorphRecon");

    cl::Program& program = cache.getProgram(sources, morphReconParams);

    morphReconInitScan<MARKER_TYPE, MASK_TYPE>(marker, mask,
                                               width, height, queue, program);
}

/**
 * @brief Second step in morphological reconstruction. Queue is initialized.
 *
 * @param marker     Buffer with marker
 * @param mask       Buffer with mask
 * @param queueData  Buffer for queue, should be size of marker and mask
 * @param width      Marker and mask width
 * @param height     Marker and mask height
 * @param queue_size Final queue occupancy is stored here
 * @param queue      Queue used to execute kernels
 * @param program    Program from which kernels are obtained
 */
template <typename MARKER_TYPE, typename MASK_TYPE>
void morphReconInitQueue(cl::Buffer marker, cl::Buffer mask,
                    cl::Buffer queueData,
                    int width, int height, int& queue_size,
                    cl::CommandQueue &queue, cl::Program &program)
{
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

    /* obtaining kernel */
    cl::Kernel init_queue_kernel(program, "init_queue_kernel");

    /* allocating buffers */

    queue_size = 0;

    cl::Buffer queue_size_buff(context, CL_TRUE, sizeof(int));
    queue.enqueueWriteBuffer(queue_size_buff, CL_TRUE, 0,
                           sizeof(int), &queue_size);

    init_queue_kernel.setArg(0, marker);
    init_queue_kernel.setArg(1, mask);
    init_queue_kernel.setArg(2, queueData);
    init_queue_kernel.setArg(3, queue_size_buff);
    init_queue_kernel.setArg(4, width);
    init_queue_kernel.setArg(5, height);

    /* calculating kernel dimensions */

    cl_int status;

    {
        int local_x = 64;
        int local_y = 4;

        cl::NDRange local(local_x, local_y);

        int global_x = ((width + local_x - 1) / local_x) * local_x;
        int global_y = ((height + local_y - 1) / local_y) * local_y;

        cl::NDRange global(global_x, global_y);

#ifdef DEBUG_PRINT
        std::cout << "global_x: " << global_x
                  <<", global_y: " << global_y << std::endl;
#endif

        status = queue.enqueueNDRangeKernel(init_queue_kernel,
                                            cl::NullRange, global, local);
    }

    queue.enqueueReadBuffer(queue_size_buff, CL_TRUE, 0,
                            sizeof(int), &queue_size);
}

/**
 * @brief See: morphReconInitQueue(cl::Buffer, cl::Buffer, cl::Buffer,
 *                             int, int, int&, cl::CommandQueue&, cl::Program&);
 */
template <typename MARKER_TYPE, typename MASK_TYPE>
void morphReconInitQueue(cl::Buffer marker, cl::Buffer mask,
                    cl::Buffer queueData,
                    int width, int height, int& queue_size,
                    ProgramCache& cache = ProgramCache::getGlobalInstance(),
                    cl::CommandQueue& queue = ProgramCache::getDefaultCommandQueue())
{
    std::cout << "parallel queue-based morphological reconstruction, init queue "
                 "ocl program params: " << morphReconParams << std::endl;

    std::vector<std::string> sources;
    sources.push_back("ParallelQueue");
    sources.push_back("MorphRecon");

    cl::Program& program = cache.getProgram(sources, morphReconParams
                              + morphReconTypeParams<MARKER_TYPE, MASK_TYPE>());

    morphReconInitQueue<MARKER_TYPE, MASK_TYPE>(marker, mask, queueData,
                                    width, height, queue_size, queue, program);
}

template <typename MARKER_TYPE, typename MASK_TYPE>
void morphReconQueuePropagation(cl::Buffer queue_data,
                      std::vector<int>& data_elements,
                      std::vector<int>& queues_sizes, int blocks_num,
                      cl::Buffer marker, cl::Buffer mask, int width, int height,
                      cl::CommandQueue &queue, cl::Program &program)
{
    assert(data_elements.size() == queues_sizes.size());
    assert(data_elements.size() == blocks_num);

    cl::Buffer device_queue_metadata;

    // initialize queue metadata buffer
    initQueueMetadata(data_elements, queues_sizes,
                      device_queue_metadata, queue);

    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

    cl::Kernel morph_recon_kernel(program, "morph_recon_kernel");

    cl::Buffer device_total_elements(context, CL_TRUE,
                                     blocks_num * sizeof(int));

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
    morph_recon_kernel.setArg(1, marker);
    morph_recon_kernel.setArg(2, mask);
    morph_recon_kernel.setArg(3, width);
    morph_recon_kernel.setArg(4, height);
    morph_recon_kernel.setArg(5, queue_data);
    morph_recon_kernel.setArg(6, device_queue_metadata);
    morph_recon_kernel.setArg(7, local_queue);
    morph_recon_kernel.setArg(8, reduction_buffer);
    morph_recon_kernel.setArg(9, got_work);
    morph_recon_kernel.setArg(10, prefix_sum_input);
    morph_recon_kernel.setArg(11, prefix_sum_output);

    cl::NDRange global(QUEUE_NUM_THREADS * blocks_num, 1);
    cl::NDRange local(QUEUE_NUM_THREADS, 1);

#ifdef DEBUG_PRINT
    std::cout << "running morphological reconstruction kernel..."
              << std::endl;
#endif

    queue.enqueueNDRangeKernel(morph_recon_kernel,
                               cl::NullRange, global, local);
}

/**
 * @brief Third, main step of morphological reconstruction. Algorithms works
 *        using parallel queue.
 *
 * @param queue_data    Buffer with queue initialized in previous step
 * @param data_elements Number of items in the queue
 * @param queue_size    Total size of queue
 * @param marker        Buffer with marker
 * @param mask          Buffer with mask
 * @param width         Marker and mask width
 * @param height        Marker and mask height
 * @param queue         Queue used to execute kernels
 * @param program       Program from which kernels are obtained
 */
template <typename MARKER_TYPE, typename MASK_TYPE>
void morphReconQueuePropagation(cl::Buffer queue_data, int data_elements,
                int queue_size,
                cl::Buffer marker, cl::Buffer mask, int width, int height,
                cl::CommandQueue &queue, cl::Program &program)
{
    std::vector<int> data_elements_vec;
    std::vector<int> queue_sizes;

    data_elements_vec.push_back(data_elements);
    queue_sizes.push_back(queue_size);

    morphReconQueuePropagation<MARKER_TYPE, MASK_TYPE>(
                queue_data, data_elements_vec, queue_sizes,
                1, marker, mask, width, height, queue, program);

}

template <typename MARKER_TYPE, typename MASK_TYPE>
void morphReconQueuePropagation(cl::Buffer queue_data, int data_elements,
                int queue_size, cl::Buffer marker, cl::Buffer mask,
                int width, int height,
                ProgramCache& cache = ProgramCache::getGlobalInstance(),
                cl::CommandQueue& queue = ProgramCache::getDefaultCommandQueue())
{
    std::vector<std::string> sources;
    sources.push_back("ParallelQueue");
    sources.push_back("MorphRecon");

    cl::Program& program = cache.getProgram(sources, morphReconParams);

    morphReconQueuePropagation<MARKER_TYPE, MASK_TYPE>(queue_data,
       data_elements, queue_size, marker, mask, width, height, queue, program);
}



/**
 * @brief Entry point for queue-based morphological reconstruction
 *
 * @param marker    Buffer containing grayscale marker (elements of size int)
 * @param mask      Buffer containing grayscale mask (elements of size int)
 * @param width     Marker and mask width
 * @param height    Marker and mask height
 * @param cache     Program cache from which ocl programs are obtained.
 *                  By default global cache instance is used.
 * @param queue     Queue used to execute kernels. By default global instance
 *                  is used.
 */
template <typename MARKER_TYPE, typename MASK_TYPE>
void morphRecon(cl::Buffer marker, cl::Buffer mask, int width, int height,
                int raster_scans_num, int blocks_num,
                ProgramCache& cache = ProgramCache::getGlobalInstance(),
                cl::CommandQueue& queue = ProgramCache::getDefaultCommandQueue())
{
    std::vector<std::string> sources;
    sources.push_back("ParallelQueue");
    sources.push_back("MorphRecon");

    cl::Program& program = cache.getProgram(sources,
            morphReconParams + morphReconTypeParams<MARKER_TYPE, MASK_TYPE>());

    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

    for(int i = 0; i < raster_scans_num; ++i)
    {
        morphReconInitScan<MARKER_TYPE, MASK_TYPE>
                (marker, mask, width, height, queue, program);
    }

    int queue_total_size = width * height;
    int queue_data_size;

    cl::Buffer device_queue_data(context, CL_TRUE,
                                 sizeof(int) * queue_total_size);

    morphReconInitQueue<MARKER_TYPE, MASK_TYPE>(marker, mask, device_queue_data,
                                                width, height, queue_data_size,
                                                queue, program);

    std::cout << "queue initialization finished, queue init size:"
              << queue_data_size << std::endl;


    int single_queue_chunk = (queue_data_size + blocks_num - 1) / blocks_num;

    // multiplication by 2 because is is size of input and output queue
    int single_queue_size = single_queue_chunk * QUEUE_EXPAND_FACTOR * 2;
    int total_queues_size = single_queue_size * blocks_num;

    // allocating device queues
    cl::Buffer device_queues(context, CL_TRUE, sizeof(int) * total_queues_size);

    std::vector<int> nums_of_elements;
    std::vector<int> total_sizes;

    int src_offset = 0;
    int dst_offset = 0;

    // partitioning data and putting to consecutive queues

    for(int i = 0; i < blocks_num; ++i)
    {
        total_sizes.push_back(single_queue_size);

        int current_chunk;

        if(i + 1 == blocks_num)
            current_chunk = queue_data_size - i * single_queue_chunk;
        else
            current_chunk = single_queue_chunk;

        nums_of_elements.push_back(current_chunk);

        queue.enqueueCopyBuffer(device_queue_data, device_queues,
                                src_offset * sizeof(int),
                                dst_offset * sizeof(int),
                                current_chunk * sizeof(int));

        src_offset += current_chunk;
        dst_offset += single_queue_size;
    }

    morphReconQueuePropagation<MARKER_TYPE, MASK_TYPE>(
                device_queues, nums_of_elements, total_sizes, blocks_num,
                marker, mask, width, height, queue, program);
}


//void morphRecon(cl::Buffer marker, cl::Buffer mask,
//                int width, int height,
//                ProgramCache& cache = ProgramCache::getGlobalInstance(),
//                cl::CommandQueue& queue = ProgramCache::getGlobalInstance()
//                                                    .getDefaultCommandQueue())
//{
//    std::string types = "-DMARKER_TYPE="
//            + TypeResolver<MARKER_TYPE>::type_as_string
//            + " -DMASK_TYPE=" + Ty

//    morphRecon(marker, mask, types, width, height, cache, queue);
//}



#endif // MORPH_RECON_H
