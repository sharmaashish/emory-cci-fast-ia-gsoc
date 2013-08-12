#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "TestUtils.h"

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cfloat>
#include <cassert>
#include <cmath>

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

/* output sum should have lenght equal to iterations,
 * iteration shouldn't be greater than num of threads in block
 */
__global__ void partial_sum_test(int* output_sum, int iterations)
{
    setCurrentQueue(blockIdx.x, blockIdx.x);

    int loopIt = 0;
    int workUnit = -1;
    int tid = threadIdx.x;

    // SHARED MEMORY FOR PARALLEL REDUCTION
    __shared__ int reduction_buffer[SUM_TEST_BLOCK_SIZE];

    int partial_sum;

    for(int i = 0; i < iterations; ++i)
    {
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

        if(tid == i)
            partial_sum = reduction_buffer[0];
    }

    // WRITING FINAL RESULT TO GLOBAL MEMORY
    if(tid < iterations)
        output_sum[blockIdx.x * iterations + tid] = partial_sum;
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

    int *host_dequeueVector = new int[queueInitDataSize];

    if(host_dequeueVector == NULL)
        std::cout << "malloc failed!" << std::endl;
   
    checkError(cudaMemcpy((void*)host_dequeueVector, (const void*)device_dequeueVector, sizeof(int) * queueInitDataSize, cudaMemcpyDeviceToHost));

    for(int i = 0;i < queueInitDataSize; ++i)
    {
        std::cout << "i[" << i << "]: " << host_dequeueVector[i] << std::endl;
    }

    checkError(cudaFree(device_queueInitData));
    checkError(cudaFree(device_outVector));
    checkError(cudaFree(device_dequeueVector));

    delete[] host_dequeueVector;
}

BOOST_AUTO_TEST_CASE(partialSum)
{
    std::cout << "GLOBAL QUEUE - PARALLEL PARTIAL SUM TEST" << std::endl;

#ifdef PREFIX_SUM
    std::cout << "PREFIX SUM ENABLED" << std::endl;
#endif

    const int queueInitDataSize = 14096;
    const int numberOfIterations = 32;

    int host_queueInitData[queueInitDataSize];

    /*random values from range <0;9> */

    for(int i = 0; i < queueInitDataSize; ++i)
    {
        host_queueInitData[i] = rand() % 10;
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

    checkError(cudaMalloc(&device_outputSum, numberOfIterations * sizeof(int)));

    partial_sum_test<<<1, SUM_TEST_BLOCK_SIZE>>>(device_outputSum, numberOfIterations);

    lastError();

    int host_outputSum[numberOfIterations];

    checkError(cudaMemcpy((void*)(host_outputSum), (void*)device_outputSum,
				numberOfIterations * sizeof(int), cudaMemcpyDeviceToHost));


    int verificationData[numberOfIterations];

    for(int i = 0, j = 0; i < numberOfIterations; ++i)
    {
        int sum = 0;

        for(int k = 0; k < SUM_TEST_BLOCK_SIZE; ++k)
        {
            if(j < queueInitDataSize)
                sum += host_queueInitData[j++];
            else
                break;
        }

        verificationData[i] = sum;
    }


    std::cout << "output partial sums: " << std::endl;

    for(int i = 0; i < numberOfIterations; ++i)
    {
        std::cout << "outputSum[" << i << "]: " << host_outputSum[i]
                     << ", cpu: " << verificationData[i]
                        << (host_outputSum[i] == verificationData[i] ? ", OK" : ", FAIL")
                        << std::endl;
    }

    checkError(cudaFree(device_queueInitData));
    checkError(cudaFree(device_outVector));
    checkError(cudaFree(device_outputSum));
}

BOOST_AUTO_TEST_CASE(partialSumMultipleBlocks)
{
    std::cout << "GLOBAL QUEUE - PARALLEL PARTIAL SUM MULTIPLE BLOCKS TEST"
              << std::endl;

#ifdef PREFIX_SUM
    std::cout << "PREFIX SUM ENABLED" << std::endl;
#endif

    const int queueInitDataSize = 1024 * 128; // 512KB
    const int numberOfIterations = 4;
    const int numberOfBlocks = 64;
    const int itemsPerBlock = queueInitDataSize / numberOfBlocks;

    int host_queueInitData[queueInitDataSize];

    /*random values from range <0;9> */

    for(int i = 0; i < queueInitDataSize; ++i)
    {
        host_queueInitData[i] = rand() % 10;
    }

    int *device_queueInitData;
    unsigned int initDataSizeByte = queueInitDataSize * sizeof(int);

    checkError(cudaMalloc(&device_queueInitData, initDataSizeByte));
    checkError(cudaMemcpy(device_queueInitData, host_queueInitData,
               initDataSizeByte, cudaMemcpyHostToDevice));

    int *device_outVector;
    checkError(cudaMalloc(&device_outVector, queueInitDataSize * sizeof(int)));

    int* host_queueInitInPointers[numberOfBlocks];
    int* host_queueInitOutPointers[numberOfBlocks];

    int host_queueSizes[numberOfBlocks];

    for(int i = 0; i < numberOfBlocks; ++i)
    {
        host_queueInitInPointers[i] = device_queueInitData + i * itemsPerBlock;
        host_queueInitOutPointers[i] = device_outVector + i * itemsPerBlock;

        host_queueSizes[i] = itemsPerBlock;
    }

    int** device_queueInitInPointers;
    int** device_queueInitOutPointers;

    checkError(cudaMalloc(&device_queueInitInPointers, numberOfBlocks * sizeof(int*)));
    checkError(cudaMemcpy(device_queueInitInPointers, host_queueInitInPointers,
               numberOfBlocks * sizeof(int*), cudaMemcpyHostToDevice));

    checkError(cudaMalloc(&device_queueInitOutPointers, numberOfBlocks * sizeof(int*)));
    checkError(cudaMemcpy(device_queueInitOutPointers, host_queueInitOutPointers,
               numberOfBlocks * sizeof(int*), cudaMemcpyHostToDevice));

    //out sizes (the same for in/out)
    int *device_queueSizes;
    checkError(cudaMalloc(&device_queueSizes, numberOfBlocks * sizeof(int)));
    checkError(cudaMemcpy(device_queueSizes, host_queueSizes,
               numberOfBlocks * sizeof(int), cudaMemcpyHostToDevice));


    //initQueue<<<1, 1>>>(device_queueInitData, queueInitDataSize,
    //                    device_outVector, queueInitDataSize);

    initQueueVector<<<1, numberOfBlocks>>>(device_queueInitInPointers, device_queueSizes,
                                           device_queueInitOutPointers, device_queueSizes,
                                           numberOfBlocks);

    lastError();

    int *device_outputSum;

    checkError(cudaMalloc(&device_outputSum, numberOfIterations * numberOfBlocks * sizeof(int)));

    partial_sum_test<<<numberOfBlocks, SUM_TEST_BLOCK_SIZE>>>(device_outputSum, numberOfIterations);

    lastError();

    int host_outputSum[numberOfIterations * numberOfBlocks];

    checkError(cudaMemcpy((void*)(host_outputSum), (void*)device_outputSum,
                numberOfIterations * numberOfBlocks * sizeof(int), cudaMemcpyDeviceToHost));


    int verificationData[numberOfIterations * numberOfBlocks];

    for(int i = 0, j = 0; i < numberOfIterations * numberOfBlocks; ++i)
    {
        int sum = 0;

        for(int k = 0; k < SUM_TEST_BLOCK_SIZE; ++k)
        {
            if(j < queueInitDataSize)
                sum += host_queueInitData[j++];
            else
                break;
        }

        verificationData[i] = sum;
    }


    std::cout << "output partial sums: " << std::endl;

    for(int i = 0; i < numberOfIterations * numberOfBlocks; ++i)
    {
        std::cout << "outputSum[" << i << "]: " << host_outputSum[i]
                     << ", cpu: " << verificationData[i]
                        << (host_outputSum[i] == verificationData[i] ? ", OK" : ", FAIL")
                        << std::endl;
    }

    checkError(cudaFree(device_queueInitData));
    checkError(cudaFree(device_outVector));
    checkError(cudaFree(device_queueInitInPointers));
    checkError(cudaFree(device_queueInitOutPointers));
    checkError(cudaFree(device_queueSizes));
    checkError(cudaFree(device_outputSum));
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

    checkError(cudaMemcpy((void*)(&host_outputSum),
                          (const void*)device_outputSum,sizeof(int),
                          cudaMemcpyDeviceToHost));

	std::cout << "output sum: " << host_outputSum << std::endl;

    checkError(cudaFree(device_queueInitData));
    checkError(cudaFree(device_outVector));
    checkError(cudaFree(device_outputSum));
}

BOOST_AUTO_TEST_CASE(morphReconstruction)
{
    const unsigned char host_image[] = {
        //       4           8           12          16          20          24          28          32
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 /*4*/  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0,
 /*8*/  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
 /*12*/ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 /*16*/ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 /*20*/ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 /*24*/ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 /*28*/ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 /*32*/ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    int host_seeds[] = { /* single '1' on 9x9 */
//               4           8           12          16          20          24          28          32
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 /*4*/  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 /*8*/  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 /*12*/ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 /*16*/ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 /*20*/ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 /*24*/ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 /*28*/ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 /*32*/ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };


    int host_image_size = sizeof(host_image) / sizeof(host_image[0]);
    int host_seeds_size = sizeof(host_seeds) / sizeof(host_seeds[0]);

    (void)host_image_size; //unused, to avoid warnings
    (void)host_seeds_size; //unused, to avoid warnings

    int ncols = 32;
    int nrows = 32;

    int total_size = ncols * nrows;

    assert(host_image_size == host_seeds_size);
    assert(total_size == host_image_size);

    const int host_input_list[] = {8*32+8, 4*32+24};
    int data_elements = sizeof(host_input_list) / sizeof(host_input_list[0]);

    std::cout << "data_elements: " << data_elements << std::endl;

    for(int i = 0; i < data_elements; ++i){
        std::cout << "input_list[" << i << "]: " << host_input_list[i]
                     << "(" << host_seeds[host_input_list[i]] << ")"
                     << std::endl;
    }

    int *device_input_list;
    cudaMalloc((void **)&device_input_list, sizeof(host_input_list)) ;
    checkError(cudaMemcpy(device_input_list, host_input_list,
                          sizeof(host_input_list), cudaMemcpyHostToDevice));

    unsigned char* device_image;
    int* device_seeds;

    checkError(cudaMalloc(&device_image, total_size * sizeof(unsigned char)));
    checkError(cudaMemcpy(device_image, host_image,
               total_size * sizeof(unsigned char), cudaMemcpyHostToDevice));

    checkError(cudaMalloc(&device_seeds, total_size * sizeof(int)));
    checkError(cudaMemcpy(device_seeds, host_seeds,
               total_size * sizeof(int), cudaMemcpyHostToDevice));

	std::cout << "running morphRecon" << std::endl;

    morphRecon(device_input_list, data_elements, device_seeds,
               device_image, ncols, nrows);

    checkError(cudaMemcpy(host_seeds, device_seeds,
               total_size * sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "printing output..." << std::endl;

    for(int i = 0; i < 32; ++i)
    {
        for(int j = 0; j < 32; ++j)
        {
            std::cout << host_seeds[i*32 + j];
        }
        std::cout << std::endl;
    }

    checkError(cudaFree(device_input_list));
    checkError(cudaFree(device_image));
    checkError(cudaFree(device_seeds));

}
