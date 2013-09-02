/*
    QUEUE_MAX_NUM_BLOCKS, QUEUE_NUM_THREADS <-- should be passed when program is build
*/
//#define DEBUG

#ifdef DEBUG
#define assert(x) \
            if (! (x)) \
            { \
                printf((__constant char*)"Assert(%s) failed in line: %d\n", \
                       (__constant char*)#x, __LINE__); \
            }
#else
        #define assert(X)
#endif

#define METADATA_PER_BLOCK_SIZE 10

#define IN_QUEUE_SIZE(BLOCK_IDX) \
   *((__global int*)(queue_metadata + METADATA_PER_BLOCK_SIZE * BLOCK_IDX + 0))
#define IN_QUEUE_OFFSET(BLOCK_IDX) \
   *((__global int*)(queue_metadata + METADATA_PER_BLOCK_SIZE * BLOCK_IDX + 1))
#define IN_QUEUE_HEAD(BLOCK_IDX) \
   *((__global int*)(queue_metadata + METADATA_PER_BLOCK_SIZE * BLOCK_IDX + 2))
#define OUT_QUEUE_MAX_SIZE(BLOCK_IDX) \
   *((__global int*)(queue_metadata + METADATA_PER_BLOCK_SIZE * BLOCK_IDX + 3))
#define OUT_QUEUE_HEAD(BLOCK_IDX) \
   *((__global int*)(queue_metadata + METADATA_PER_BLOCK_SIZE * BLOCK_IDX + 4))
#define OUT_QUEUE_OFFSET(BLOCK_IDX) \
   *((__global int*)(queue_metadata + METADATA_PER_BLOCK_SIZE * BLOCK_IDX + 5))
#define CURR_IN_QUEUE_OFFSET(BLOCK_IDX) \
   *((__global int*)(queue_metadata + METADATA_PER_BLOCK_SIZE * BLOCK_IDX + 6))
#define CURR_OUT_QUEUE_OFFSET(BLOCK_IDX) \
   *((__global int*)(queue_metadata + METADATA_PER_BLOCK_SIZE * BLOCK_IDX + 7))
#define TOTAL_INSERTS(BLOCK_IDX) \
   *((__global int*)(queue_metadata + METADATA_PER_BLOCK_SIZE * BLOCK_IDX + 8))
#define EXECUTION_CODE(BLOCK_IDX) \
   *((__global int*)(queue_metadata + METADATA_PER_BLOCK_SIZE * BLOCK_IDX + 9))

#define EXECUTION_CODE_GLOBAL \
   *((__global int*)(queue_metadata + METADATA_PER_BLOCK_SIZE * get_num_groups(0)))

//#define QUEUE_WORKSPACE       __global char* queue_workspace
//#define QUEUE_WORKSPACE_ARG   queue_workspace

#define QUEUE_DATA            __global int* queue_data
#define QUEUE_METADATA        __global int* queue_metadata


// Makes queue 1 point to queue 2, and vice-versa
void swapQueues(QUEUE_METADATA, int loopIt){

    barrier(CLK_GLOBAL_MEM_FENCE);

    int group_id = get_group_id(0);
    int local_id = get_local_id(0);

    if(loopIt %2 == 0)
    {
        CURR_IN_QUEUE_OFFSET(group_id) = OUT_QUEUE_OFFSET(group_id);
        CURR_OUT_QUEUE_OFFSET(group_id) = IN_QUEUE_OFFSET(group_id);

        if(local_id == 0)
        {
            IN_QUEUE_SIZE(group_id) = OUT_QUEUE_HEAD(group_id);
            OUT_QUEUE_HEAD(group_id) = 0;
            IN_QUEUE_HEAD(group_id) = 0;
            // This is used for profiling only
            TOTAL_INSERTS(group_id) += IN_QUEUE_SIZE(group_id);
        }
    }
    else
    {
        CURR_IN_QUEUE_OFFSET(group_id) = IN_QUEUE_OFFSET(group_id);
        CURR_OUT_QUEUE_OFFSET(group_id) = OUT_QUEUE_OFFSET(group_id);

        if(local_id == 0)
        {
            IN_QUEUE_SIZE(group_id) = OUT_QUEUE_HEAD(group_id);
            OUT_QUEUE_HEAD(group_id) = 0;
            IN_QUEUE_HEAD(group_id) = 0;
            // This is used for profiling only
            TOTAL_INSERTS(group_id) += IN_QUEUE_SIZE(group_id);
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
}


// -2, nothing else to be done at all
int dequeueElement(QUEUE_DATA, QUEUE_METADATA, int *loopIt, __local volatile int* gotWork)
{
    int threadIdx = get_local_id(0);
    int blockIdx = get_group_id(0);

getWork:
        *gotWork = 0;

        // Try to get some work.
        int queue_index = IN_QUEUE_HEAD(blockIdx) + threadIdx;

        barrier(CLK_LOCAL_MEM_FENCE);

        if(get_local_id(0) == 0){
            IN_QUEUE_HEAD(blockIdx) += get_local_size(0);
        }

        // Nothing to do by default
        int element = -1;
        if(queue_index < IN_QUEUE_SIZE(blockIdx)){
            element = queue_data[CURR_IN_QUEUE_OFFSET(blockIdx) + queue_index];
            *gotWork = 1;
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

        // This block does not have anything to process
        if(!*gotWork){
            element = -2;
            if(OUT_QUEUE_HEAD(blockIdx) != 0){
                swapQueues(queue_metadata, loopIt[0]);
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

            assert(ai < QUEUE_NUM_THREADS);
            assert(bi < QUEUE_NUM_THREADS);

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

            assert(ai < QUEUE_NUM_THREADS);
            assert(bi < QUEUE_NUM_THREADS);

            int t = prefix_sum_output[ai];
            prefix_sum_output[ai] = prefix_sum_output[bi];
            prefix_sum_output[bi] += t;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}


// Assuming that all threads in a block are calling this function
// prefix sum input, prefix_sum_output - size should be equal to QUEUE_NUM_THREADS
int queueElement(QUEUE_DATA, QUEUE_METADATA,
                 __local int *elements,
                 __local int* prefix_sum_input,
                 __local int* prefix_sum_output)
{
    //printf("queue!\n");

    int threadIdx = get_local_id(0);
    int blockIdx = get_group_id(0);

    int global_queue_index = OUT_QUEUE_HEAD(blockIdx);

    // set to the number of values this threard is writing
    prefix_sum_input[threadIdx] = elements[0];

    // run a prefix-sum on threads inserting data to the queue
    scan(prefix_sum_input, prefix_sum_output);

    // calculate index into the queue where given thread is writing
    int queue_index = global_queue_index + prefix_sum_output[threadIdx];

    for(int i = 0; i < elements[0]; i++)
    {
        // If the queue storage has been exceed, than set the execution code to 1.
        // This will force a second round in the morphological reconstruction.
        if(queue_index + i >= OUT_QUEUE_MAX_SIZE(blockIdx))
        {
             EXECUTION_CODE(blockIdx) = 1;
             EXECUTION_CODE_GLOBAL = 1;
        }
        else
        {
            queue_data[CURR_OUT_QUEUE_OFFSET(blockIdx) + queue_index + i]
                                                            = elements[i + 1];
        }
    }

    // thread 0 updates head of the queue
    if(threadIdx == 0)
    {
        OUT_QUEUE_HEAD(blockIdx) += prefix_sum_output[QUEUE_NUM_THREADS-1]
                                  + prefix_sum_input[QUEUE_NUM_THREADS-1];

        if(OUT_QUEUE_HEAD(blockIdx) >= OUT_QUEUE_MAX_SIZE(blockIdx))
        {
            OUT_QUEUE_HEAD(blockIdx) = OUT_QUEUE_MAX_SIZE(blockIdx);
            //printf("max exceeded");
        }
    }
    return queue_index;
}
