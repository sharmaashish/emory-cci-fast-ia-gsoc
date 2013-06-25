//template <typename T>
//__global__ void replaceKernel(int rows, int cols, const PtrStep_<T> img1, PtrStep_<T> result, T oldval, T newval)
//{
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;

//    if (y < rows && x < cols)
//    {
//    	T p = img1.ptr(y)[x];
//    	result.ptr(y)[x] = (p == oldval ? newval : p);
//    }
//}


__kernel void replace(__global uchar *src, int src_pitch,
                      __global uchar *dst, int dst_pitch,
                      int width, int height, uchar oldval, uchar newval)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < width && y < height)
    {
        int src_index = y * src_pitch + x;
        int dst_index = y * dst_pitch + x;

      	uchar p = src[src_index];
       	dst[x] = (p == oldval ? newval : p);
    }
}
