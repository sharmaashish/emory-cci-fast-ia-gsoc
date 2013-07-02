#include "watershed.h"
#include "utils/ocl_utils.h"

#include <iostream>

//__constant__ int N_xs[8] = {-1,0,1,1,1,0,-1,-1};
//__constant__ int N_ys[8] = {-1,-1,-1,0,1,1,1,0};

static const char neighbourhood_x[] = {-1, 0, 1, 1, 1, 0,-1,-1};
static const char neighbourhood_y[] = {-1,-1,-1, 0, 1, 1, 1, 0};

void watershed(cl::CommandQueue& queue, cl::Kernel& kernel,
            int width, int height,
            cl::Buffer& src,// int src_pitch,
            cl::Buffer& labeled)
{
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

    //setting constant memory with neigbourhood
    cl::Buffer cl_neighbourhood_x = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(neighbourhood_x));
    cl::Buffer cl_neighbourhood_y = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(neighbourhood_y));

    queue.enqueueWriteBuffer(cl_neighbourhood_x, CL_TRUE, 0, sizeof(neighbourhood_x), neighbourhood_x);
    queue.enqueueWriteBuffer(cl_neighbourhood_y, CL_TRUE, 0, sizeof(neighbourhood_y), neighbourhood_y);

    const size_t block_size = 6;
    cl::LocalSpaceArg local_mem = cl::__local(block_size * block_size);

    //setting args for descent_kernel
    kernel.setArg(0, src);
    kernel.setArg(1, labeled);
    kernel.setArg(2, cl_neighbourhood_x);
    kernel.setArg(3, cl_neighbourhood_y);
    kernel.setArg(4, local_mem);
    kernel.setArg(5, width);
    kernel.setArg(6, height);

    size_t global_width = width / (block_size - 2) * block_size;
    size_t global_height = height / (block_size - 2) * block_size;

    cl::NDRange NullRange;
    cl::NDRange global(global_width, global_height);
    cl::NDRange local(block_size, block_size);

    queue.enqueueNDRangeKernel(kernel, NullRange, global, local);
}
