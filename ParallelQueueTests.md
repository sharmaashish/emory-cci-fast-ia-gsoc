# Details #

  * Insert some elements, then pop them. Compare the popped set to the insert set
  * Put the queue in a loop, insert, compute sum of the entries in the queue, insert, compute sum of the entries, repeat... and plot the sum values for the parallel queue and a serial version
  * To test the FIFO property, modify 2 so that we push one, pop one at each iteration, the sum is of what's in the queue.  Plot as well.

To write tests for CUDA code, some changes in project structure has been made. Starting from compute capability 2.0 nvcc compiler supports separable compilation.
To take advantage of this feature, appropriate changes in cmake configuration has been done.

Parallel queue implementation is in file 'global\_queue.cu'. To use code from this file in tests, header 'global\_queue.cuh' has been created.

## test 1 ##

Very simple test, takes items from queue and copy to given pointer.

Invocation for this kernel can be found in 'initializeAndDequeue' boost test case (globalQueueTest.cu).

```
__global__ void dequeue_test(int* device_result)
{
    setCurrentQueue(blockIdx.x, blockIdx.x);

    int loopIt = 0;
    int workUnit = dequeueElement(&loopIt);

    device_result[blockIdx.x * blockDim.x + threadIdx.x] = workUnit;
}
```

## test 2 ##

This kernel performs parallel reduction on data obtained from queue.
After each iteration, sum of obtained elements is added to the queue.

Finally, kernel writes value from last reduction to given pointer (output\_sum).

Invocation for this kernel can be found in 'sum' boost test case (globalQueueTest.cu).

use case:
  * init queue with 2048 elements, all of them equal to 1
  * set number of iterations to 5
  * set number of threads to 512
  * the output obtained from kernel (via given pointer) should be equal to 2048


```
__global__ void sum_test(int* output_sum, int iterations)
{
    setCurrentQueue(blockIdx.x, blockIdx.x);

    int loopIt = 0;
    int workUnit = -1;
    int tid = threadIdx.x;
    
    __shared__ int localQueue[SUM_TEST_BLOCK_SIZE][2];

    // SHARED MEMORY FOR PARALLEL REDUCTION
    __shared__ int reduction_buffer[SUM_TEST_BLOCK_SIZE];

    for(int i = 0; i < iterations; ++i)
    {
        localQueue[tid][0] = 0;

        // Try to get some work.
        workUnit = dequeueElement(&loopIt);
		
        // PREPARING NEXT DATA PART FROM QUEUE TO REDUCTION
        reduction_buffer[tid] = (workUnit < 0 ? 0 : workUnit);
		
        __syncthreads();

        // SIMPLE REDUCTION
        for (unsigned int s = blockDim.x/2; s > 0; s >>= 1)
        {
            if (tid < s) {
                reduction_buffer[tid] += reduction_buffer[tid + s];
            }
            __syncthreads();
        }

        // PUTTING SUM TO LOCAL QUEUE
        if(tid == 0)
        {
            localQueue[tid][0] = 1;
            localQueue[tid][1] = reduction_buffer[0];
        }

        // PUTTING SUM TO GLOBAL QUEUE
        if(i != iterations - 1)
            queueElement(localQueue[tid]);
    }

    // WRITING FINAL RESULT TO GLOBAL MEMORY
    if(tid == 0)
        *output_sum = reduction_buffer[0];
} 

```

## test 3 ##