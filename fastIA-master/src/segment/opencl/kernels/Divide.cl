//template <typename T>
//void divideCaller(int rows, int cols, const PtrStep_<T> img1,
//		const PtrStep_<T> img2, PtrStep_<T> result, cudaStream_t stream)
//{
//    dim3 threads(16, 16);
//    dim3 grid((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

//    divideKernel<<<grid, threads, 0, stream>>>(rows, cols, img1, img2, result);
//     cudaGetLastError() ;

//    if (stream == 0)
//        cudaDeviceSynchronize();
//}


__kernel void divide(__global uchar *src_1, int src_1_pitch,
	             __global uchar *src_2, int src_2_pitch,
                     __global uchar *dst, int dst_pitch,
                     int width, int height)
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
        dst[dst_index] = p / q;
    }
}
