#ifndef MORPH_RECON_H
#define MORPH_RECON_H

#include <CL/cl.hpp>
#include "utils/ocl_program_cache.h"
#include "utils/ocl_type_resolver.h"

#include "parallel_queue.h"

#include <sstream>
#include <cassert>
#include <vector>

//#define DEBUG_PRINT

/* This flag causes error on GOTO on AMD*/
//#define WARNINGS_AS_ERRORS

#define INIT_SCAN_X_THREADS_X 16
#define INIT_SCAN_X_THREADS_Y 16

#define INIT_SCAN_Y_THREAD_X 256

#define QUEUE_EXPAND_FACTOR 2

#include "morph_recon_internal.h"

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

    int execution_code;

    do
    {
        for(int i = 0; i < raster_scans_num; ++i)
        {
            morphReconInitScan<MARKER_TYPE, MASK_TYPE>
                    (marker, mask, width, height, queue, program);
        }

        int queue_total_size = width * height;
        int queue_data_size;

        cl::Buffer device_queue_data(context, CL_TRUE,
                                     sizeof(int) * queue_total_size);

        morphReconInitQueue(marker, mask, device_queue_data,
                            width, height, queue_data_size,
                            queue, program);

    #ifdef DEBUG_PRINT
        std::cout << "queue initialization finished, queue init size:"
                  << queue_data_size << std::endl;
    #endif

        if(!queue_data_size)
            break;

        int single_queue_chunk = (queue_data_size + blocks_num - 1) / blocks_num;

        // multiplication by 2 because is is size of input and output queue
        int single_queue_size = single_queue_chunk * QUEUE_EXPAND_FACTOR * 2;
        int total_queues_size = single_queue_size * blocks_num;

        // allocating device queues
        cl::Buffer device_queues(context, CL_TRUE,
                                 sizeof(int) * total_queues_size);

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

        morphReconQueuePropagation(
                device_queues, nums_of_elements, total_sizes, blocks_num,
                marker, mask, width, height, execution_code, queue, program);

#ifdef DEBUG_PRINT
        std::cout << "execution code:" << execution_code << std::endl;
#endif

    }while(execution_code);
}


#endif // MORPH_RECON_H
