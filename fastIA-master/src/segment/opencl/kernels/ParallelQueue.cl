#define QUEUE_MAX_NUM_BLOCKS	70
#define QUEUE_NUM_THREADS	512

__global static volatile int inQueueSize[QUEUE_MAX_NUM_BLOCKS];
__global static volatile int *inQueuePtr1[QUEUE_MAX_NUM_BLOCKS];
__global static volatile int inQueueHead[QUEUE_MAX_NUM_BLOCKS];
__global static volatile int outQueueMaxSize[QUEUE_MAX_NUM_BLOCKS];
__global static volatile int outQueueHead[QUEUE_MAX_NUM_BLOCKS];
__global static volatile int *outQueuePtr2[QUEUE_MAX_NUM_BLOCKS];

__global static volatile int *curInQueue[QUEUE_MAX_NUM_BLOCKS];
__global static volatile int *curOutQueue[QUEUE_MAX_NUM_BLOCKS];
__global static volatile int execution_code;


// This variables are used for debugging purposes only
__global static volatile int totalInserts[QUEUE_MAX_NUM_BLOCKS];


void setCurrentQueue(int currentQueueIdx, int queueIdx)
{
        curInQueue[currentQueueIdx] = inQueuePtr1[queueIdx];
        curOutQueue[currentQueueIdx] = outQueuePtr2[queueIdx];
}

// Makes queue 1 point to queue 2, and vice-versa
void swapQueues(int loopIt){

      //  printf("swapping queues\n");

        barrier(CLK_LOCAL_MEM_FENCE);

        int group_id = get_group_id(0);
        int local_id = get_local_id(0);

        if(loopIt %2 == 0){
                curInQueue[group_id] = outQueuePtr2[group_id];
                curOutQueue[group_id] = inQueuePtr1[group_id];
                if(local_id == 0){
                        inQueueSize[group_id] = outQueueHead[group_id];
                        outQueueHead[group_id] = 0;
                        inQueueHead[group_id] = 0;
                        // This is used for profiling only
                        totalInserts[group_id] += inQueueSize[group_id];
                }
        }else{
                curInQueue[group_id] = inQueuePtr1[group_id];
                curOutQueue[group_id] = outQueuePtr2[group_id];

                if(local_id == 0){
                        inQueueSize[group_id] = outQueueHead[group_id];
                        outQueueHead[group_id] = 0;
                        inQueueHead[group_id] = 0;
                        // This is used for profiling only
                        totalInserts[group_id] += inQueueSize[group_id];
                }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
}


// -2, nothing else to be done at all
int dequeueElement(int *loopIt, __local volatile int* gotWork){
        // did this block got something to do?
getWork:
        *gotWork = 0;


        // Try to get some work.
//	int queue_index = atomicAdd((int*)&inQueueHead, 1);
        int queue_index = inQueueHead[get_group_id(0)] + get_local_id(0);
        // I must guarantee that idle threads are set to 0, and no other thread
        // will come later and set it to 0 again
        //__syncthreads();

        barrier(CLK_LOCAL_MEM_FENCE);

        if(get_local_id(0) == 0){
                inQueueHead[get_group_id(0)] += get_local_size(0);
//		if(loopIt[0] < 1){
//			printf("inQueueSize = %d loopIt[0] = %d queue_index = %d outQueueHead = %d\n", inQueueSize[blockIdx.x], loopIt[0], queue_index, outQueueHead[blockIdx.x]);
//		}
        }

        // Nothing to do by default
    int element = -1;
        if(queue_index < inQueueSize[get_group_id(0)]){
                element = curInQueue[get_group_id(0)][queue_index];
                *gotWork = 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);


        // This block does not have anything to process
        if(!*gotWork){
//		if(loopIt[0] < 20 && threadIdx.x == 0)
//			printf("inQueueSize = %d loopIt[0] = %d\n", inQueueSize[blockIdx.x], loopIt[0]);
                element = -2;
                if(outQueueHead[get_group_id(0)] != 0){
                        swapQueues(loopIt[0]);
                        loopIt[0]++;
                        goto getWork;
                }
        }
        return element;
}

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


// Utils...
// http://www.moderngpu.com/intro/scan.html

// prefix_sum_workspace_1 size: QUEUE_NUM_WARPS * QUEUE_SCAN_STRIDE
// prefix_sum_workspace_2 size: QUEUE_NUM_WARPS + QUEUE_NUM_WARPS / 2
void scan_old(__local const int* prefix_sum_input,
          __local int* prefix_sum_output,
          __local volatile int* prefix_sum_workspace_1,
          __local volatile int* prefix_sum_workspace_2)
{
    int tid = get_local_id(0);
    int size = get_local_size(0);

    if(tid == 0)
    {
        prefix_sum_output[0] = 0;
        for(int i = 1; i < size; ++i)
        {
            prefix_sum_output[i] = prefix_sum_output[i-1] + prefix_sum_input[i-1];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

// values -> prefix_sum_input
// exclusive -> prefix_sum_output

        // Reserve a half warp of extra space plus one per warp in the block.
        // This is exactly enough space to avoid comparisons in the multiscan
        // and to avoid bank conflicts.
    //__shared__ volatile int scan[QUEUE_NUM_WARPS * QUEUE_SCAN_STRIDE];

    // scan -> prefix_sum_workspace_1
/*
    int tid = get_local_id(0);
    int warp = tid / QUEUE_WARP_SIZE;
    int lane = (QUEUE_WARP_SIZE - 1) & tid;

    __local volatile int* s = prefix_sum_workspace_1 + QUEUE_SCAN_STRIDE * warp + lane + QUEUE_WARP_SIZE / 2;
        s[-16] = 0;

        // Read from global memory.
        int x = prefix_sum_input[tid];
        s[0] = x;

        // Run inclusive scan on each warp's data.
        int sum = x;

//#pragma unroll
        for(int i = 0; i < 5; ++i) {
                int offset = 1<< i;
                sum += s[-offset];
                s[0] = sum;
        }

        // Synchronize to make all the totals available to the reduction code.
        //__syncthreads();
        barrier(CLK_LOCAL_MEM_FENCE);

    //__shared__ volatile int totals[QUEUE_NUM_WARPS + QUEUE_NUM_WARPS / 2];

    if(tid < QUEUE_NUM_WARPS) {
                // Grab the block total for the tid'th block. This is the last element
                // in the block's scanned sequence. This operation avoids bank
                // conflicts.
        int total = prefix_sum_workspace_1[QUEUE_SCAN_STRIDE * tid + QUEUE_WARP_SIZE / 2 + QUEUE_WARP_SIZE - 1];

                prefix_sum_workspace_2[tid] = 0;
        __local volatile int* s2 = prefix_sum_workspace_2 + QUEUE_NUM_WARPS / 2 + tid;
                int totalsSum = total;
                s2[0] = total;

//#pragma unroll
        for(int i = 0; i < LOG_QUEUE_NUM_WARPS; ++i) {
                        int offset = 1<< i;
                        totalsSum += s2[-offset];
                        s2[0] = totalsSum;
                }

                // Subtract total from totalsSum for an exclusive scan.
                prefix_sum_workspace_2[tid] = totalsSum - total;
        }

        // Synchronize to make the block scan available to all warps.
        //__syncthreads();
        barrier(CLK_LOCAL_MEM_FENCE);

        // Add the block scan to the inclusive sum for the block.
        sum += prefix_sum_workspace_2[warp];

        // Write the inclusive and exclusive scans to global memory.
//	inclusive[tid] = sum;
        prefix_sum_output[tid] = sum - x;*/
}

// Assuming that all threads in a block are calling this function
// prefix sum input, prefix_sum_output - size should be equal to QUEUE_NUM_THREADS
int queueElement(__local int *elements,
                 __local int* prefix_sum_input,
                 __local int* prefix_sum_output)
{
    int queue_index = -1;

    int threadIdx = get_local_id(0);
    int blockIdx = get_group_id(0);

    //__shared__ int writeAddr[QUEUE_NUM_THREADS];
    //__shared__ int exclusiveScan[QUEUE_NUM_THREADS];

    //__shared__
    int global_queue_index;

    //if(threadIdx.x == 0){
    global_queue_index = outQueueHead[blockIdx];
    //}

    // set to the number of values this threard is writing
    prefix_sum_input[threadIdx] = elements[0];

    // run a prefix-sum on threads inserting data to the queue
    scan(prefix_sum_input, prefix_sum_output);

    // calculate index into the queue where given thread is writing
    queue_index = global_queue_index + prefix_sum_output[threadIdx];

    // write elemets sequentially to shared memory
//	int localIndex = exclusiveScan[threadIdx.x];
//	for(int i = 0; i < elements[0]; i++){
//		localElements[localIndex+i] = elements[i+1];
//	}

//	__syncthreads();
//	for(int i = threadIdx.x; i < exclusiveScan[QUEUE_NUM_THREADS-1]+writeAddr[QUEUE_NUM_THREADS-1]; i+=blockDim.x){
//		curOutQueue[blockIdx.x][global_queue_index+i] = localElements[i];
//	}

    for(int i = 0; i < elements[0]; i++){
            // If the queue storage has been exceed, than set the execution code to 1.
            // This will force a second round in the morphological reconstructio.
            if(queue_index + i >= outQueueMaxSize[blockIdx]){
//			printf("List out of bounds\n");
                    execution_code=1;
            }else{
                    curOutQueue[blockIdx][queue_index + i] = elements[i + 1];
                    printf("queue: %d\n", elements[i + 1]);
            }
    }

    // thread 0 updates head of the queue
    if(threadIdx == 0)
    {
        printf("moving head: %d + %d\n", prefix_sum_output[QUEUE_NUM_THREADS-1], prefix_sum_input[QUEUE_NUM_THREADS-1]);
        outQueueHead[blockIdx]+=prefix_sum_output[QUEUE_NUM_THREADS-1] + prefix_sum_input[QUEUE_NUM_THREADS-1];

        if(outQueueHead[blockIdx] >= outQueueMaxSize[blockIdx])
        {
            outQueueHead[blockIdx] = outQueueMaxSize[blockIdx];
        }
//		printf("Inserting = %d - outQueueHead = %d\n", exclusiveScan[QUEUE_NUM_THREADS-1]+writeAddr[QUEUE_NUM_THREADS-1], outQueueHead[blockIdx.x]);
    }
    return queue_index;
}



__kernel void init_queue_kernel(__global int* inQueueData, int dataElements,
                                __global int* outQueueData, int outMaxSize)
{
    if(get_global_id(0) < 1){
        // Simply assign input data pointers/number of elements to the queue
        inQueuePtr1[0] = inQueueData;

        //printf("initQueueVector: tid - %d dataElements = %d pointer = %p\n",
        //    threadIdx.x, dataElements, inQueueData);
        inQueueSize[0] = dataElements;

        totalInserts[0] = 0;

        //alloc second vector used to queue output elements
        outQueuePtr2[0] = outQueueData;

        //Maximum number of elements that fit into the queue
        outQueueMaxSize[0] = outMaxSize;

        //Head of the out queue
        outQueueHead[0] = 0;

        //Head of the in queue
        inQueueHead[0] = 0;
    }
}


__kernel void dequeue_test(__global int* device_result, __local int* gotWork)
{
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int group_size = get_local_size(0);

    setCurrentQueue(group_id, group_id);

    int loopIt = 0;
    int workUnit = dequeueElement(&loopIt, gotWork);

    device_result[group_id * group_size + local_id] = workUnit;
}

// reduction_buffer size should be size of block
__kernel void partial_sum_test(__global int* output_sum, int iterations,
                               __local int* reduction_buffer,
                               __local int* gotWork)
{

    int blockIdx = get_group_id(0);
    int blockDim = get_local_size(0);
    int tid = get_local_id(0);

    setCurrentQueue(blockIdx, blockIdx);

    int loopIt = 0;
    int workUnit = -1;
    //int tid = threadIdx.x;

    // SHARED MEMORY FOR PARALLEL REDUCTION
    //__shared__ int reduction_buffer[SUM_TEST_BLOCK_SIZE];

    int partial_sum;

    for(int i = 0; i < iterations; ++i)
    {
        // Try to get some work.
        workUnit = dequeueElement(&loopIt, gotWork);

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


__kernel void sum_test(__global int* output_sum, int iterations,
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

    setCurrentQueue(blockIdx, blockIdx);

    int loopIt = 0;
    int workUnit = -1;

        //__shared__ int localQueue[SUM_TEST_BLOCK_SIZE][2];

        // SHARED MEMORY FOR PARALLEL REDUCTION
        //__shared__ int reduction_buffer[SUM_TEST_BLOCK_SIZE];

    for(int i = 0; i < iterations; ++i)
    {
        //localQueue[tid][0] = 0;
        local_queue[tid * 2] = 0;

        // Try to get some work.
        workUnit = dequeueElement(&loopIt, gotWork);

        if(!tid){
            printf("first work unit: %d\n", workUnit);
        }

        // PREPARING NEXT DATA PART FROM QUEUE TO REDUCTION
        reduction_buffer[tid] = (workUnit < 0 ? 0 : workUnit);

        //__syncthreads();
        barrier(CLK_LOCAL_MEM_FENCE);

        // SIMPLE REDUCTION
        for (unsigned int s = blockDim/2; s > 0; s >>= 1)
        {
                if (tid < s) {
                        reduction_buffer[tid] += reduction_buffer[tid + s];
                }
                //__syncthreads();
                barrier(CLK_LOCAL_MEM_FENCE);
        }

        // PUTTING SUM TO LOCAL QUEUE
        if(tid == 0)
        {
            local_queue[tid * 2] = 1;
            local_queue[tid * 2 + 1] = reduction_buffer[0];
            printf("storing reduction: %d\n", reduction_buffer[0]);
        }

        // PUTTING SUM TO GLOBAL QUEUE
        if(i != iterations - 1)
                queueElement(local_queue + tid*2, prefix_sum_input,
                    prefix_sum_output);
    }

    // WRITING FINAL RESULT TO GLOBAL MEMORY
    if(tid == 0)
            *output_sum = reduction_buffer[0];
}
