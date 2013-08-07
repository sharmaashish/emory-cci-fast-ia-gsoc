/*
    QUEUE_MAX_NUM_BLOCKS, QUEUE_NUM_THREADS <-- should be passed when program is build
*/

#define IN_QUEUE_SIZE_OFFSET      (0)
#define IN_QUEUE_PTR_1_OFFSET     (sizeof(int) * QUEUE_MAX_NUM_BLOCKS)
#define IN_QUEUE_HEAD_OFFSET      (sizeof(int*) * QUEUE_MAX_NUM_BLOCKS + IN_QUEUE_PTR_1_OFFSET)
#define OUT_QUEUE_MAX_SIZE_OFFSET (sizeof(int) * QUEUE_MAX_NUM_BLOCKS + IN_QUEUE_HEAD_OFFSET)
#define OUT_QUEUE_HEAD_OFFSET     (sizeof(int) * QUEUE_MAX_NUM_BLOCKS + OUT_QUEUE_MAX_SIZE_OFFSET)
#define OUT_QUEUE_PTR_2_OFFSET    (sizeof(int) * QUEUE_MAX_NUM_BLOCKS + OUT_QUEUE_HEAD_OFFSET)
#define CUR_IN_QUEUE_OFFSET       (sizeof(int*) * QUEUE_MAX_NUM_BLOCKS + OUT_QUEUE_PTR_2_OFFSET)
#define CUR_OUT_QUEUE_OFFSET      (sizeof(int*) * QUEUE_MAX_NUM_BLOCKS + CUR_IN_QUEUE_OFFSET)
#define EXECUTION_CODE_OFFSET     (sizeof(int*) * QUEUE_MAX_NUM_BLOCKS + CUR_OUT_QUEUE_OFFSET)
#define TOTAL_INSERTS_OFFSET      (sizeof(int) + EXECUTION_CODE_OFFSET)

#define IN_QUEUE_SIZE         ((__global int*)(queue_workspace + IN_QUEUE_SIZE_OFFSET))
#define IN_QUEUE_PTR_1        ((__global int* __global*)(queue_workspace + IN_QUEUE_PTR_1_OFFSET))
#define IN_QUEUE_HEAD         ((__global int*)(queue_workspace + IN_QUEUE_HEAD_OFFSET))
#define OUT_QUEUE_MAX_SIZE    ((__global int*)(queue_workspace + OUT_QUEUE_MAX_SIZE_OFFSET))
#define OUT_QUEUE_HEAD        ((__global int*)(queue_workspace + OUT_QUEUE_HEAD_OFFSET))
#define OUT_QUEUE_PTR_2       ((__global int* __global*)(queue_workspace + OUT_QUEUE_PTR_2_OFFSET))
#define CUR_IN_QUEUE          ((__global int* __global*)(queue_workspace + CUR_IN_QUEUE_OFFSET))
#define CUR_OUT_QUEUE         ((__global int* __global*)(queue_workspace + CUR_OUT_QUEUE_OFFSET))
#define EXECUTION_CODE        (((__global int*)(queue_workspace + EXECUTION_CODE_OFFSET))[0])
#define TOTAL_INSERTS         ((__global int*)(queue_workspace + TOTAL_INSERTS_OFFSET))

#define QUEUE_WORKSPACE       __global char* queue_workspace
#define QUEUE_WORKSPACE_ARG   queue_workspace

void setCurrentQueue(QUEUE_WORKSPACE, int currentQueueIdx, int queueIdx)
{
        CUR_IN_QUEUE[currentQueueIdx] = IN_QUEUE_PTR_1[queueIdx];
        CUR_OUT_QUEUE[currentQueueIdx] = OUT_QUEUE_PTR_2[queueIdx];
}

// Makes queue 1 point to queue 2, and vice-versa
void swapQueues(QUEUE_WORKSPACE, int loopIt){

    barrier(CLK_LOCAL_MEM_FENCE);

    int group_id = get_group_id(0);
    int local_id = get_local_id(0);

    if(loopIt %2 == 0)
    {
        CUR_IN_QUEUE[group_id] = OUT_QUEUE_PTR_2[group_id];
        CUR_OUT_QUEUE[group_id] = IN_QUEUE_PTR_1[group_id];
        if(local_id == 0)
        {
            IN_QUEUE_SIZE[group_id] = OUT_QUEUE_HEAD[group_id];
            OUT_QUEUE_HEAD[group_id] = 0;
            IN_QUEUE_HEAD[group_id] = 0;
            // This is used for profiling only
            TOTAL_INSERTS[group_id] += IN_QUEUE_SIZE[group_id];
        }
    }
    else
    {
        CUR_IN_QUEUE[group_id] = IN_QUEUE_PTR_1[group_id];
        CUR_OUT_QUEUE[group_id] = OUT_QUEUE_PTR_2[group_id];

        if(local_id == 0)
        {
            IN_QUEUE_SIZE[group_id] = OUT_QUEUE_HEAD[group_id];
            OUT_QUEUE_HEAD[group_id] = 0;
            IN_QUEUE_HEAD[group_id] = 0;
            // This is used for profiling only
            TOTAL_INSERTS[group_id] += IN_QUEUE_SIZE[group_id];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}


// -2, nothing else to be done at all
int dequeueElement(QUEUE_WORKSPACE, int *loopIt, __local volatile int* gotWork){

getWork:
        *gotWork = 0;

        // Try to get some work.
        int queue_index = IN_QUEUE_HEAD[get_group_id(0)] + get_local_id(0);

        barrier(CLK_LOCAL_MEM_FENCE);

        if(get_local_id(0) == 0){
            IN_QUEUE_HEAD[get_group_id(0)] += get_local_size(0);
        }

        // Nothing to do by default
        int element = -1;
        if(queue_index < IN_QUEUE_SIZE[get_group_id(0)]){
            element = CUR_IN_QUEUE[get_group_id(0)][queue_index];
            *gotWork = 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // This block does not have anything to process
        if(!*gotWork){
            element = -2;
            if(OUT_QUEUE_HEAD[get_group_id(0)] != 0){
                swapQueues(QUEUE_WORKSPACE_ARG, loopIt[0]);
                loopIt[0]++;
                goto getWork;
            }
        }
        return element;
}

// perform exclusive prefix sum
void scan(__local const int* prefix_sum_input,
          __local int* prefix_sum_output)
{
    int tid = get_local_id(0);
    int size = get_local_size(0);

    prefix_sum_output[tid] = prefix_sum_input[tid];

    int offset = 1; //exclusive scan

    for (int d = size>>1; d > 0; d >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < d)
        {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;
            prefix_sum_output[bi] += prefix_sum_output[ai];
        }
        offset *= 2;
    }

    if (tid == 0) prefix_sum_output[size - 1] = 0;

    for (int d = 1; d < size; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < d)
        {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;
            int t = prefix_sum_output[ai];
            prefix_sum_output[ai] = prefix_sum_output[bi];
            prefix_sum_output[bi] += t;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}


// Assuming that all threads in a block are calling this function
// prefix sum input, prefix_sum_output - size should be equal to QUEUE_NUM_THREADS
int queueElement(QUEUE_WORKSPACE,
                 __local int *elements,
                 __local int* prefix_sum_input,
                 __local int* prefix_sum_output)
{
    int queue_index = -1;

    int threadIdx = get_local_id(0);
    int blockIdx = get_group_id(0);

    int global_queue_index = OUT_QUEUE_HEAD[blockIdx];

    // set to the number of values this threard is writing
    prefix_sum_input[threadIdx] = elements[0];

    // run a prefix-sum on threads inserting data to the queue
    scan(prefix_sum_input, prefix_sum_output);

    // calculate index into the queue where given thread is writing
    queue_index = global_queue_index + prefix_sum_output[threadIdx];

    for(int i = 0; i < elements[0]; i++)
    {
        // If the queue storage has been exceed, than set the execution code to 1.
        // This will force a second round in the morphological reconstructio.
        if(queue_index + i >= OUT_QUEUE_MAX_SIZE[blockIdx])
            EXECUTION_CODE=1;
        else
            CUR_OUT_QUEUE[blockIdx][queue_index + i] = elements[i + 1];
    }

    // thread 0 updates head of the queue
    if(threadIdx == 0)
    {
        OUT_QUEUE_HEAD[blockIdx] += prefix_sum_output[QUEUE_NUM_THREADS-1]
                                  + prefix_sum_input[QUEUE_NUM_THREADS-1];

        if(OUT_QUEUE_HEAD[blockIdx] >= OUT_QUEUE_MAX_SIZE[blockIdx])
            OUT_QUEUE_HEAD[blockIdx] = OUT_QUEUE_MAX_SIZE[blockIdx];
    }
    return queue_index;
}

__kernel void init_queue_kernel(QUEUE_WORKSPACE,
                                __global int* inQueueData, int dataElements,
                                __global int* outQueueData, int outMaxSize)
{
    if(get_global_id(0) < 1)
    {
        // Simply assign input data pointers/number of elements to the queue
        IN_QUEUE_PTR_1[0] = inQueueData;

        IN_QUEUE_SIZE[0] = dataElements;

        TOTAL_INSERTS[0] = 0;

        //alloc second vector used to queue output elements
        OUT_QUEUE_PTR_2[0] = outQueueData;

        //Maximum number of elements that fit into the queue
        OUT_QUEUE_MAX_SIZE[0] = outMaxSize;

        //Head of the out queue
        OUT_QUEUE_HEAD[0] = 0;

        //Head of the in queue
        IN_QUEUE_HEAD[0] = 0;
    }
}

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
