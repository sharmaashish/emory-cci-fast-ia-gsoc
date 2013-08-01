#ifndef GLOBAL_QUEUE_CUH
#define GLOBAL_QUEUE_CUH

#define QUEUE_MAX_NUM_BLOCKS	70

#define QUEUE_WARP_SIZE 	32
#define QUEUE_NUM_THREADS	512
#define QUEUE_NUM_WARPS (QUEUE_NUM_THREADS / QUEUE_WARP_SIZE)
#define LOG_QUEUE_NUM_THREADS 9
#define LOG_QUEUE_NUM_WARPS (LOG_QUEUE_NUM_THREADS - 5)

#define QUEUE_SCAN_STRIDE (QUEUE_WARP_SIZE + QUEUE_WARP_SIZE / 2 + 1)

__device__ void scan(const int* values, int* exclusive);
__device__ int queueElement(int *outQueueCurPtr, int *elements);
__device__ int queueElement(int *elements);
__device__ int queueElement(int element);
__device__ void swapQueues(int loopIt);
__device__ int dequeueElement(int *loopIt);

__device__ int propagate(int *seeds, unsigned char *image,
                         int x, int y, int ncols, unsigned char pval);

__device__ void setCurrentQueue(int currentQueueIdx, int queueIdx);

__global__ void initQueue(int *inQueueData, int dataElements,
                          int *outQueueData, int outMaxSize);

__global__ void initQueueId(int *inQueueData, int dataElements,
                            int *outQueueData, int outMaxSize, int qId);

// added by M. Cieslak
__global__ void initQueueVector(int **inQueueData, int *inQueueSizes,
                                int **outQueueData, int *outQueueSizes, int numImages);

__global__ void initQueueVector(int **inQueueData, int *inQueueSizes,
                                int **outQueueData, int numImages);

/* host functions */

extern "C" int listComputation(int *h_Data, int dataElements,
                               int *d_seeds, unsigned char *d_image, int ncols, int nrows);

extern "C" int morphReconVector(int nImages, int **h_InputListPtr,
                                int* h_ListSize, int **h_Seeds, unsigned char **h_Images,
                                int* h_ncols, int* h_nrows, int connectivity);

extern "C" int morphReconSpeedup(int *g_InputListPtr, int h_ListSize,
                                 int *g_Seed, unsigned char *g_Image, int h_ncols, int h_nrows,
                                 int connectivity, int nBlocks, float queue_increase_factor);

extern "C" int morphRecon(int *d_input_list, int dataElements,
                          int *d_seeds, unsigned char *d_image, int ncols, int nrows);

#endif //GLOBAL_QUEUE_CUH
