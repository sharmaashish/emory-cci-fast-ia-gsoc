//__global__ void bgr2grayKernel(int rows, int cols, const PtrStep_<unsigned char> img,
// 	PtrStep_<unsigned char> result)
//{
//	int x = blockIdx.x * blockDim.x + threadIdx.x;
//	int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//	// Same constants as used by Matlab
//	double r_const = 0.298936021293776;
//	double g_const = 0.587043074451121;
//	double b_const = 0.114020904255103;
//
//	if (y < rows && x < cols)
//	{
//		unsigned char b = img.ptr(y)[ x * 3];
//		unsigned char g = img.ptr(y)[ x * 3 + 1];
//		unsigned char r = img.ptr(y)[ x * 3 + 2];
//		double grayPixelValue =  r_const * (double)r + g_const * (double)g + b_const * (double)b;
//		result.ptr(y)[x] = double2char(grayPixelValue);
//	}
//}

#pragma OPENCL EXTENSION cl_khr_fp64: enable

uchar double2char(double d){
	double truncate = min( max(d, (double )0.0), (double)255.0);
	double pt;
	double c = modf(truncate, &pt) >= .5?ceil(truncate):floor(truncate);
	return (uchar)c;
}


__kernel void bgr2gray(__global uchar *src, int src_pitch,
                       __global uchar *dst, int dst_pitch,
                       int width, int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    double r_const = 0.298936021293776;
    double g_const = 0.587043074451121;
    double b_const = 0.114020904255103;

    if(x < width && y < height)
    {
        int src_index = y * src_pitch + x * 3;
        int dst_index = y * dst_pitch + x;    

	uchar b = src[src_index];
	uchar g = src[src_index + 1];
	uchar r = src[src_index + 2];
	
	double grayPixelValue =  r_const * (double)r + g_const * (double)g + b_const * (double)b;
	
	dst[dst_index] = double2char(grayPixelValue);
    }
}

