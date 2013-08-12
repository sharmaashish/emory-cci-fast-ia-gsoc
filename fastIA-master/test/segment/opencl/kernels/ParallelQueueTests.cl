
__kernel void dequeue_test(QUEUE_WORKSPACE,
                        __global int* device_result,
                        __local int* gotWork)
{
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int group_size = get_local_size(0);

    setCurrentQueue(QUEUE_WORKSPACE_ARG, group_id, group_id);

    int loopIt = 0;
    int workUnit = dequeueElement(QUEUE_WORKSPACE_ARG, &loopIt, gotWork);

    device_result[group_id * group_size + local_id] = workUnit;
}


// reduction_buffer size should be size of block
__kernel void partial_sum_test(QUEUE_WORKSPACE,
                               __global int* output_sum, int iterations,
                               __local int* reduction_buffer,
                               __local int* gotWork)
{
    int blockIdx = get_group_id(0);
    int blockDim = get_local_size(0);
    int tid = get_local_id(0);

    setCurrentQueue(QUEUE_WORKSPACE_ARG, blockIdx, blockIdx);

    int loopIt = 0;
    int workUnit = -1;
    //int tid = threadIdx.x;

    int partial_sum;

    for(int i = 0; i < iterations; ++i)
    {
        // Try to get some work.
        workUnit = dequeueElement(QUEUE_WORKSPACE_ARG, &loopIt, gotWork);

        // PREPARING NEXT DATA PART FROM QUEUE TO REDUCTION
        reduction_buffer[tid] = (workUnit < 0 ? 0 : workUnit);

        barrier(CLK_LOCAL_MEM_FENCE);

        // SIMPLE REDUCTION
        for (unsigned int s = blockDim/2; s > 0; s >>= 1)
        {
            if (tid < s) {
                reduction_buffer[tid] += reduction_buffer[tid + s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if(tid == i)
            partial_sum = reduction_buffer[0];
    }

    // WRITING FINAL RESULT TO GLOBAL MEMORY
    if(tid < iterations)
        output_sum[blockIdx * iterations + tid] = partial_sum;
}

__kernel void sum_test(QUEUE_WORKSPACE,
                       __global int* output_sum, int iterations,
                       __local int *local_queue,
                       __local int *reduction_buffer,
                       __local int* gotWork,
                       // queue stuff:
                       __local int* prefix_sum_input,
                       __local int* prefix_sum_output)
{

    int blockIdx = get_group_id(0);
    int blockDim = get_local_size(0);
    int tid = get_local_id(0);

    setCurrentQueue(QUEUE_WORKSPACE_ARG, blockIdx, blockIdx);

    int loopIt = 0;
    int workUnit = -1;

    for(int i = 0; i < iterations; ++i)
    {
        //localQueue[tid][0] = 0;
        local_queue[tid * 2] = 0;

        // Try to get some work.
        workUnit = dequeueElement(QUEUE_WORKSPACE_ARG, &loopIt, gotWork);

        // PREPARING NEXT DATA PART FROM QUEUE TO REDUCTION
        reduction_buffer[tid] = (workUnit < 0 ? 0 : workUnit);

        barrier(CLK_LOCAL_MEM_FENCE);

        // SIMPLE REDUCTION
        for (unsigned int s = blockDim/2; s > 0; s >>= 1)
        {
                if (tid < s) {
                        reduction_buffer[tid] += reduction_buffer[tid + s];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
        }

        // PUTTING SUM TO LOCAL QUEUE
        if(tid == 0)
        {
            local_queue[tid * 2] = 1;
            local_queue[tid * 2 + 1] = reduction_buffer[0];
        }

        // PUTTING SUM TO GLOBAL QUEUE
        if(i != iterations - 1)
                queueElement(QUEUE_WORKSPACE_ARG, local_queue + tid*2, prefix_sum_input,
                    prefix_sum_output);
    }

    // WRITING FINAL RESULT TO GLOBAL MEMORY
    if(tid == 0)
            *output_sum = reduction_buffer[0];
}

