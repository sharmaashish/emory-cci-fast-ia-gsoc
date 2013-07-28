#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "TestUtils.h"

#include <stdlib.h>
#include <stdio.h>
#include <cfloat>
#include <assert.h>

#include <cuda_runtime_api.h>

#include "global_queue.cuh"

// created for gpu testing purposes
// compilation: nvcc DoNothingKernel.cu -o DoNothingKernel

__global__ void do_nothing_kernel(unsigned char* input, int size,
                                  int num_of_iters){

    for(int j = 0; j < num_of_iters; ++j){
        for(int i = 0; i < size; ++i){
            input[i] = sqrtf(input[i]) + sqrtf(input[size - 1 - i]);
        }
    }
}


__global__ void my_listReduceKernel(int* d_Result, int *seeds, unsigned char *image, int ncols, int nrows){
//    curInQueue[blockIdx.x] = inQueuePtr1[blockIdx.x];
//    curOutQueue[blockIdx.x] = outQueuePtr2[blockIdx.x];

    int loopIt = 0;
    int workUnit = -1;
    int tid = threadIdx.x;
    __shared__ int localQueue[QUEUE_NUM_THREADS][5];

    do{
        int x, y;

        localQueue[tid][0] = 0;

        // Try to get some work.
        workUnit = dequeueElement(&loopIt);
        y = workUnit/ncols;
        x = workUnit%ncols;

        unsigned char pval = 0;

        if(workUnit >= 0){
            pval = seeds[workUnit];
        }

        int retWork = -1;
        if(workUnit > 0){
            retWork = propagate((int*)seeds, image, x, y-1, ncols, pval);
            if(retWork > 0){
                localQueue[tid][0]++;
                localQueue[tid][localQueue[tid][0]] = retWork;
            }
        }
//		queueElement(retWork);
        if(workUnit > 0){
            retWork = propagate((int*)seeds, image, x, y+1, ncols, pval);
            if(retWork > 0){
                localQueue[tid][0]++;
                localQueue[tid][localQueue[tid][0]] = retWork;
            }
        }
//		queueElement(retWork);

        if(workUnit > 0){
            retWork = propagate((int*)seeds, image, x-1, y, ncols, pval);
            if(retWork > 0){
                localQueue[tid][0]++;
                localQueue[tid][localQueue[tid][0]] = retWork;
            }
        }
//		queueElement(retWork);

        if(workUnit > 0){
            retWork = propagate((int*)seeds, image, x+1, y, ncols, pval);
            if(retWork > 0){
                localQueue[tid][0]++;
                localQueue[tid][localQueue[tid][0]] = retWork;
            }
        }
        queueElement(localQueue[tid]);

    }while(workUnit != -2);

//    d_Result[0]=totalInserts[blockIdx.x];
}

int my_listComputation(int *h_Data, int dataElements, int *d_seeds, unsigned char *d_image, int ncols, int nrows){
// seeds contais the maker and it is also the output image

//	uint threadsX = 512;
    int blockNum = 1;
    int *d_Result;

    int *d_Data;
    unsigned int dataSize = dataElements * sizeof(int);
    cudaMalloc((void **)&d_Data, dataSize  );
    cudaMemcpy(d_Data, h_Data, dataSize, cudaMemcpyHostToDevice);

    // alloc space to save output elements in the queue
    int *d_OutVector;
    cudaMalloc((void **)&d_OutVector, sizeof(int) * dataElements);

//	printf("Init queue data!\n");
    // init values of the __global__ variables used by the queue
    initQueue<<<1, 1>>>(d_Data, dataElements, d_OutVector, dataElements);

//	init_sync<<<1, 1>>>();


    cudaMalloc((void **)&d_Result, sizeof(int) ) ;
    cudaMemset((void *)d_Result, 0, sizeof(int));

//	printf("Run computation kernel!\n");
    my_listReduceKernel<<<blockNum, QUEUE_NUM_THREADS>>>(d_Result, d_seeds, d_image, ncols, nrows);

//	cutilCheckMsg("histogramKernel() execution failed\n");
    int h_Result;
    cudaMemcpy(&h_Result, d_Result, sizeof(int), cudaMemcpyDeviceToHost);

    printf("	#queue entries = %d\n",h_Result);
    cudaFree(d_Data);
    cudaFree(d_Result);
    cudaFree(d_OutVector);

    // TODO: free everyone
    return h_Result;
}


BOOST_AUTO_TEST_CASE(test1)
{
    std::cout << "GLOBAL QUEUE TEST" << std::endl;

    const int queueInitDataSize = 200;
    int host_queueInitData[queueInitDataSize];

    for(int i = 0; i < queueInitDataSize; ++i)
    {
        host_queueInitData[i] = i;
    }

    int *device_queueInitData;
    unsigned int initDataSizeByte = queueInitDataSize * sizeof(int);

    cudaMalloc(&device_queueInitData, initDataSizeByte);
    cudaMemcpy(device_queueInitData, host_queueInitData,
               initDataSizeByte, cudaMemcpyHostToDevice);

    int *device_outVector;
    cudaMalloc(&device_outVector, queueInitDataSize * sizeof(int));


    initQueue<<<1, 1>>>(device_queueInitData, queueInitDataSize,
                        device_outVector, queueInitDataSize);

}


