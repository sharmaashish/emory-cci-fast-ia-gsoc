#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "TestUtils.h"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cfloat>
#include <assert.h>

//#include <cuda_runtime_api.h>

#include "global_queue.cuh"

#include "cuda/cuda_errors.h"

#define SUM_TEST_BLOCK_SIZE 512


__global__ void dequeue_test(int* device_result)
{
	setCurrentQueue(blockIdx.x, blockIdx.x);

    int loopIt = 0;
	int workUnit = dequeueElement(&loopIt);

    device_result[blockIdx.x * blockDim.x + threadIdx.x] = workUnit;
}

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



BOOST_AUTO_TEST_CASE(initializeAndDeque)
{
    std::cout << "GLOBAL QUEUE - INITIALIZE AND DEQUEUE TEST" << std::endl;

    const int queueInitDataSize = 64;
    int host_queueInitData[queueInitDataSize];

    for(int i = 0; i < queueInitDataSize; ++i)
    {
        host_queueInitData[i] = i*i;
    }

    int *device_queueInitData;
    unsigned int initDataSizeByte = queueInitDataSize * sizeof(int);

    checkError(cudaMalloc(&device_queueInitData, initDataSizeByte));
    checkError(cudaMemcpy(device_queueInitData, host_queueInitData,
               initDataSizeByte, cudaMemcpyHostToDevice));

    int *device_outVector;
    checkError(cudaMalloc(&device_outVector, queueInitDataSize * sizeof(int)));


    initQueue<<<1, 1>>>(device_queueInitData, queueInitDataSize,
                        device_outVector, queueInitDataSize);

    lastError();

    int *device_dequeueVector;

    checkError(cudaMalloc(&device_dequeueVector, queueInitDataSize * sizeof(int)));

    dequeue_test<<<1, 64>>>(device_dequeueVector);

    lastError();

    int *host_dequeueVector = (int *) malloc(queueInitDataSize * sizeof(int));

    if(host_dequeueVector == NULL)
        std::cout << "malloc failed!" << std::endl;
   
    checkError(cudaMemcpy((void*)host_dequeueVector, (const void*)device_dequeueVector, sizeof(int) * queueInitDataSize, cudaMemcpyDeviceToHost));

    for(int i = 0;i < queueInitDataSize; ++i)
    {
        std::cout << "i[" << i << "]: " << host_dequeueVector[i] << std::endl;
    }
}

BOOST_AUTO_TEST_CASE(sum)
{
    std::cout << "GLOBAL QUEUE - PARALLEL SUM TEST" << std::endl;

#ifdef PREFIX_SUM
	std::cout << "PREFIX SUM ENABLED" << std::endl;
#endif

    const int queueInitDataSize = 14096;
	const int numberOfIterations = 100;

    int host_queueInitData[queueInitDataSize];

    for(int i = 0; i < queueInitDataSize; ++i)
    {
        host_queueInitData[i] = 1;
    }

    int *device_queueInitData;
    unsigned int initDataSizeByte = queueInitDataSize * sizeof(int);

    checkError(cudaMalloc(&device_queueInitData, initDataSizeByte));
    checkError(cudaMemcpy(device_queueInitData, host_queueInitData,
               initDataSizeByte, cudaMemcpyHostToDevice));

    int *device_outVector;
    checkError(cudaMalloc(&device_outVector, queueInitDataSize * sizeof(int)));


    initQueue<<<1, 1>>>(device_queueInitData, queueInitDataSize,
                        device_outVector, queueInitDataSize);

    lastError();

    int *device_outputSum;

    checkError(cudaMalloc(&device_outputSum, sizeof(int)));

    sum_test<<<1, SUM_TEST_BLOCK_SIZE>>>(device_outputSum, numberOfIterations);

    lastError();

    int host_outputSum;

    checkError(cudaMemcpy((void*)(&host_outputSum), (const void*)device_outputSum, sizeof(int), cudaMemcpyDeviceToHost));

	std::cout << "output sum: " << host_outputSum << std::endl;
}
