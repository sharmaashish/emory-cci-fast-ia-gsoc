#include "watershed.h"
#include "utils/ocl_utils.h"

#include <iostream>

static const char neighbourhood_x[] = {-1, 0, 1, 1, 1, 0,-1,-1};
static const char neighbourhood_y[] = {-1,-1,-1, 0, 1, 1, 1, 0};

void watershed(int width, int height,
               cl::Buffer& src,
               cl::Buffer& labeled,
               ProgramCache& cache,
               cl::CommandQueue queue)

{
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

    cl::Program& program = cache.getProgram("Watershed");
    
    cl::Kernel descent_kernel(program, "descent_kernel");
    cl::Kernel increment_kernel(program, "increment_kernel");
    cl::Kernel minima_kernel(program, "minima_kernel");
    cl::Kernel plateau_kernel(program, "plateau_kernel");
    cl::Kernel flood_kernel(program, "flood_kernel");

    //setting constant memory with neigbourhood
    cl::Buffer cl_neighbourhood_x = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(neighbourhood_x));
    cl::Buffer cl_neighbourhood_y = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(neighbourhood_y));

#ifdef OPENCL_PROFILE
    cl::Event first_event;
    queue.enqueueWriteBuffer(cl_neighbourhood_x, CL_TRUE, 0, sizeof(neighbourhood_x),
                             neighbourhood_x, __null, &first_event);
#else
    queue.enqueueWriteBuffer(cl_neighbourhood_x, CL_TRUE, 0, sizeof(neighbourhood_x), neighbourhood_x);
#endif

    queue.enqueueWriteBuffer(cl_neighbourhood_y, CL_TRUE, 0, sizeof(neighbourhood_y), neighbourhood_y);

    const size_t block_size = 6;
    cl::LocalSpaceArg local_mem = cl::__local(block_size * block_size * sizeof(float));

    //setting args for descent_kernel
    descent_kernel.setArg(0, src);
    descent_kernel.setArg(1, labeled);
    descent_kernel.setArg(2, cl_neighbourhood_x);
    descent_kernel.setArg(3, cl_neighbourhood_y);
    descent_kernel.setArg(4, local_mem);
    descent_kernel.setArg(5, width);
    descent_kernel.setArg(6, height);

    size_t global_width = width / (block_size - 2) * block_size;
    size_t global_height = height / (block_size - 2) * block_size;

    cl::NDRange NullRange;
    cl::NDRange global(global_width, global_height);
    cl::NDRange local(block_size, block_size);

    cl_int status;

    status = queue.enqueueNDRangeKernel(descent_kernel, NullRange, global, local);

    std::cout << "kernel execution " << status << std::endl;

   // queue.flush();
   // queue.enqueueBarrier();

    //preparing increment kernel

    increment_kernel.setArg(0, labeled);
    increment_kernel.setArg(1, width);
    increment_kernel.setArg(2, height);

    status = queue.enqueueNDRangeKernel(increment_kernel, NullRange, global, local);
    queue.enqueueBarrier();

    //preparing minima kernel

    int counter_tmp = 0;
    cl::Buffer counter(context, CL_MEM_READ_WRITE, sizeof(int));
    queue.enqueueWriteBuffer(counter, CL_TRUE, 0, sizeof(int), &counter_tmp);

    queue.enqueueBarrier();

    minima_kernel.setArg(0, counter);
    minima_kernel.setArg(1, labeled);
    minima_kernel.setArg(2, cl_neighbourhood_x);
    minima_kernel.setArg(3, cl_neighbourhood_y);
    minima_kernel.setArg(4, local_mem);
    minima_kernel.setArg(5, width);
    minima_kernel.setArg(6, height);

    int old_val = -1;
    int new_val = -2;
    int c = 0;

    while(old_val != new_val)
    {
        old_val = new_val;
        status = queue.enqueueNDRangeKernel(minima_kernel, NullRange, global, local);
        queue.enqueueReadBuffer(counter, CL_TRUE, 0, sizeof(int), &new_val);
       // std::cout << "new_val: " << new_val << std::endl;
        c++;
    }

    std::cout << "step 2: " << c << " iterations" << std::endl;

    //preparing plateau kernel

    queue.enqueueWriteBuffer(counter, CL_TRUE, 0, sizeof(int), &counter_tmp);

    queue.enqueueBarrier();

    plateau_kernel.setArg(0, counter);
    plateau_kernel.setArg(1, src);
    plateau_kernel.setArg(2, labeled);
    plateau_kernel.setArg(3, cl_neighbourhood_x);
    plateau_kernel.setArg(4, cl_neighbourhood_y);
    plateau_kernel.setArg(5, local_mem);
    plateau_kernel.setArg(6, width);
    plateau_kernel.setArg(7, height);

    old_val = -1;
    new_val = -2;
    c = 0;

    while(old_val != new_val)
    {
        old_val = new_val;
        status = queue.enqueueNDRangeKernel(plateau_kernel, NullRange, global, local);
        queue.enqueueReadBuffer(counter, CL_TRUE, 0, sizeof(int), &new_val);
       // std::cout << "\tnew_val: " << new_val << std::endl;
        c++;
    }

    //preparing flood kernel

    queue.enqueueWriteBuffer(counter, CL_TRUE, 0, sizeof(int), &counter_tmp);
    queue.enqueueBarrier();

    flood_kernel.setArg(0, counter);
    flood_kernel.setArg(1, labeled);
    flood_kernel.setArg(2, width);
    flood_kernel.setArg(3, height);

    old_val = -1;
    new_val = -2;
    c = 0;

    int new_block_size = 16;
    local = cl::NDRange(new_block_size, new_block_size);

#ifdef OPENCL_PROFILE
    cl::Event last_event;
#endif

    while(old_val != new_val)
    {
        old_val = new_val;
        status = queue.enqueueNDRangeKernel(flood_kernel, NullRange, global, local);
#ifdef OPENCL_PROFILE
        queue.enqueueReadBuffer(counter, CL_TRUE, 0, sizeof(int), &new_val, __null, &last_event);
#else
        queue.enqueueReadBuffer(counter, CL_TRUE, 0, sizeof(int), &new_val);
#endif
 //       std::cout << "\tnew_val: " << new_val << std::endl;
        c++;
    }

#ifdef OPENCL_PROFILE
    last_event.wait();

    cl_ulong start = first_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end = last_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    cl_ulong total_time = end - start;

    setLastExecutionTime(total_time/1000000.0f);
#endif
}
