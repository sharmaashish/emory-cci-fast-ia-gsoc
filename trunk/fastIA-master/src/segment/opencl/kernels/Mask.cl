//template <typename T>
//__global__ void maskKernel(int rows, int cols, const PtrStep_<T> img1, const PtrStep_<unsigned char> img2, PtrStep_<T> result, T background)
//{
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;

//    if (y < rows && x < cols)
//    {
//    	T p = img1.ptr(y)[x];
//    	unsigned char q = img2.ptr(y)[x];
//        result.ptr(y)[x] = (q > 0) ? p : background;
//    }
//}

__kernel void mask(__global uchar *src_1, int src_1_pitch,
	           __global uchar *src_2, int src_2_pitch,
                   __global uchar *dst, int dst_pitch,
                   int width, int height, uchar background)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < width && y < height)
    {
        int src_1_index = y * src_1_pitch + x;
	int src_2_index = y * src_2_pitch + x;
        int dst_index = y * dst_pitch + x;

    	uchar p = src_1[src_1_index];
    	uchar q = src_2[src_2_index];
        dst[dst_index] = (q > 0) ? p : background;
    }
}
