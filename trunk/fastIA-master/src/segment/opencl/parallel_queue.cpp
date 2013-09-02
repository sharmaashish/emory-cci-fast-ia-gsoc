#include "parallel_queue.h"
#include "utils/ocl_utils.h"

#include <iostream>

/** Structure of 'data' buffer:
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
                       cl::Buffer& queueMetadata, cl::Buffer& execution_code,
                       cl::CommandQueue& queue)
{

    std::vector<int> dataElementsVec;
    dataElementsVec.push_back(dataElements);

    std::vector<int> totalSizes;
    totalSizes.push_back(totalSize);

    initQueueMetadata(dataElementsVec, totalSizes, queueMetadata,
                      execution_code, queue);
}

void initQueueMetadata(std::vector<int> &dataElements,
                       std::vector<int> &totalSizes,
                       cl::Buffer &queueMetadata,
                       cl::Buffer &executionCode,
                       cl::CommandQueue &queue)
{
    assert(dataElements.size() == totalSizes.size());

    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

    int size = dataElements.size();
    const int single_queue_metadata_size = 10;

    // extra element for global execution code
    int total_size = single_queue_metadata_size * size + 1;

    int* host_metadata = new int[total_size];
    int offset = 0;

    for(int i = 0; i < size; ++i)
    {
        int* ptr = host_metadata + single_queue_metadata_size * i;

        int dataElem = dataElements[i];
        int total_size = totalSizes[i];
        int total_size_half = total_size / 2;

        assert(total_size_half * 2 == total_size);
        assert(dataElem <= total_size_half);

        ptr[0] = dataElements[i];          /* input data size */
        ptr[1] = offset;                   /* input queue offset */
        ptr[2] = 0;                        /* input queue head */
        ptr[3] = total_size;               /* input/output queue max size */
        ptr[4] = 0;                        /* output queue head */
        ptr[5] = offset + total_size_half; /* output offset */
        ptr[6] = offset;                   /* current input queue offset */
        ptr[7] = offset + total_size_half; /* current output queue offset */
        ptr[8] = 0;                        /* total inserts */
        ptr[9] = 0;                        /* execution code per block */

        offset += total_size;
    }

    // setting global execution code
    host_metadata[single_queue_metadata_size * size] = 0;


    queueMetadata = cl::Buffer(context, CL_TRUE,
                      total_size * sizeof(int));

    queue.enqueueWriteBuffer(queueMetadata, CL_TRUE, 0,
                             total_size * sizeof(int), host_metadata);

    _cl_buffer_region region;
    region.origin = (total_size - 1) * sizeof(int);
    region.size = sizeof(int);

    executionCode = queueMetadata.createSubBuffer(0, CL_BUFFER_CREATE_TYPE_REGION, &region);
}



