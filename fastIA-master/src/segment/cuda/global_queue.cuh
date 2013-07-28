#ifndef GLOBAL_QUEUE_CUH
#define GLOBAL_QUEUE_CUH


#define QUEUE_MAX_NUM_BLOCKS	70
//#include "global_sync.cu"

#define QUEUE_WARP_SIZE 	32
#define QUEUE_NUM_THREADS	512
#define QUEUE_NUM_WARPS (QUEUE_NUM_THREADS / QUEUE_WARP_SIZE)
#define LOG_QUEUE_NUM_THREADS 9
#define LOG_QUEUE_NUM_WARPS (LOG_QUEUE_NUM_THREADS - 5)

#define QUEUE_SCAN_STRIDE (QUEUE_WARP_SIZE + QUEUE_WARP_SIZE / 2 + 1)

//__device__ volatile int inQueueSize[QUEUE_MAX_NUM_BLOCKS];
//__device__ volatile int *inQueuePtr1[QUEUE_MAX_NUM_BLOCKS];
//__device__ volatile int inQueueHead[QUEUE_MAX_NUM_BLOCKS];
//__device__ volatile int outQueueMaxSize[QUEUE_MAX_NUM_BLOCKS];
//__device__ volatile int outQueueHead[QUEUE_MAX_NUM_BLOCKS];
//__device__ volatile int *outQueuePtr2[QUEUE_MAX_NUM_BLOCKS];

//__device__ volatile int *curInQueue[QUEUE_MAX_NUM_BLOCKS];
//__device__ volatile int *curOutQueue[QUEUE_MAX_NUM_BLOCKS];
//__device__ volatile int execution_code;


//// This variables are used for debugging purposes only
//__device__ volatile int totalInserts[QUEUE_MAX_NUM_BLOCKS];


__device__ void scan(const int* values, int* exclusive);
__device__ int queueElement(int *outQueueCurPtr, int *elements);
__device__ int queueElement(int *elements);
__device__ int queueElement(int element);
__device__ void swapQueues(int loopIt);
__device__ int dequeueElement(int *loopIt);

__device__ int propagate(int *seeds, unsigned char *image,
                         int x, int y, int ncols, unsigned char pval);

__global__ void initQueue(int *inQueueData, int dataElements,
                          int *outQueueData, int outMaxSize);

__global__ void initQueueId(int *inQueueData, int dataElements,
                            int *outQueueData, int outMaxSize, int qId);

__global__ void initQueueVector(int **inQueueData, int *inQueueSizes,
                                int **outQueueData, int numImages);


#endif //GLOBAL_QUEUE_CUH
