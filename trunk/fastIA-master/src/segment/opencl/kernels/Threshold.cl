//template <typename T>
//__global__ void thresholdKernel(int rows, int cols, const PtrStep_<T> img1, PtrStep_<unsigned char> result, T lower, bool lower_inclusive, T upper, bool up_inclusive)
//{
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;

//    if (y < rows && x < cols)
//    {
//    	T p = img1.ptr(y)[x];
//    	bool pb = (p > lower) && (p < upper);
//    	if (lower_inclusive) pb = pb || (p == lower);
//    	if (up_inclusive) pb = pb || (p == upper);
//    	result.ptr(y)[x] = pb ? 255 : 0;
//    }
//}

__kernel void threshold(__global uchar *src, int src_pitch,
                     __global uchar *dst, int dst_pitch,
                     int width, int height,
                     uchar lower, uchar upper,
                     uchar lower_inclusive, uchar upper_inclusive)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < width && y < height)
    {
        int src_index = y * src_pitch + x;
        int dst_index = y * dst_pitch + x;

        uchar value = src[src_index];
        bool pb = (value > lower) && (value < upper);

        if (lower_inclusive) pb = pb || (value == lower);
        if (upper_inclusive) pb = pb || (value == upper);

        dst[dst_index] = pb ? 255 : 0;
    }
}

