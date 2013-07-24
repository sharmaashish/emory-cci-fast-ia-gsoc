#include "watershed.h"
#include "utils/ocl_utils.h"

#include <iostream>

static const char neighbourhood_x[] = {-1, 0, 1, 1, 1, 0,-1,-1};
static const char neighbourhood_y[] = {-1,-1,-1, 0, 1, 1, 1, 0};

//#define DEBUG_PRINT

#define TIME_DIVISOR (1000000.0f)

#ifdef OPENCL_PROFILE

float watershed_descent_kernel_time;
float watershed_increment_kernel_time;
float watershed_minima_kernel_time;
float watershed_plateau_kernel_time;
float watershed_flood_kernel_time;

int watershed_minima_kernel_iter;
int watershed_plateau_kernel_iter;
int watershed_flood_kernel_iter;

#endif


void watershed(int width, int height,
               cl::Buffer& src,
               cl::Buffer& labeled,
               ProgramCache& cache,
               cl::CommandQueue& queue)
{

#ifdef OPENCL_PROFILE
    watershed_descent_kernel_time = 0;
    watershed_increment_kernel_time = 0;
    watershed_minima_kernel_time = 0;
    watershed_plateau_kernel_time = 0;
    watershed_flood_kernel_time = 0;
#endif

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

    size_t global_width = (width / (block_size - 2) + 1) * block_size;
    size_t global_height = (height / (block_size - 2) + 1) * block_size;

#ifdef DEBUG_PRINT
    std::cout << "global width=" << global_width << " global height=" << global_height << std::endl;
#endif

    cl::NDRange NullRange;
    cl::NDRange global(global_width, global_height);
    cl::NDRange local(block_size, block_size);

    cl_int status;

#ifdef OPENCL_PROFILE
    {
        VECTOR_CLASS<cl::Event> events_vector(1);
        status = queue.enqueueNDRangeKernel(descent_kernel, NullRange, global, local, __null, &events_vector[0]);

        cl::WaitForEvents(events_vector);

        cl::Event& event = events_vector[0];

        cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        cl_ulong total_time = end - start;

        watershed_descent_kernel_time = total_time;
    }
#else
    status = queue.enqueueNDRangeKernel(descent_kernel, NullRange, global, local);
#endif

#ifdef DEBUG_PRINT
    std::cout << "kernel execution " << status << std::endl;
#endif

   // queue.flush();
   // queue.enqueueBarrier();

    /* PREPARING INCREMENT KERNEL */

    increment_kernel.setArg(0, labeled);
    increment_kernel.setArg(1, width);
    increment_kernel.setArg(2, height);

#ifdef OPENCL_PROFILE
    {
        VECTOR_CLASS<cl::Event> events_vector(1);
        status = queue.enqueueNDRangeKernel(increment_kernel, NullRange, global, local, __null, &events_vector[0]);

        cl::WaitForEvents(events_vector);

        cl::Event& event = events_vector[0];

        cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        cl_ulong total_time = end - start;

        watershed_increment_kernel_time = total_time;
    }
#else
    status = queue.enqueueNDRangeKernel(increment_kernel, NullRange, global, local);
#endif

//    queue.enqueueBarrier();


    /* PREPARING MINIMA KERNEL */

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

#ifdef OPENCL_PROFILE
        {
            VECTOR_CLASS<cl::Event> events_vector(1);
            status = queue.enqueueNDRangeKernel(minima_kernel, NullRange, global, local, __null, &events_vector[0]);

            cl::WaitForEvents(events_vector);
            cl::Event& event = events_vector[0];
            cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            cl_ulong total_time = end - start;

            watershed_minima_kernel_time += total_time;
        }
#else
        status = queue.enqueueNDRangeKernel(minima_kernel, NullRange, global, local);
#endif
        queue.enqueueReadBuffer(counter, CL_TRUE, 0, sizeof(int), &new_val);
        c++;
    }

#ifdef DEBUG_PRINT
    std::cout << "step 2: " << c << " iterations" << std::endl;
#endif

    /* PREPARING PLATEAU KERNEL */

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

#ifdef OPENCL_PROFILE
    watershed_plateau_kernel_time = 0;
#endif

    while(old_val != new_val)
    {
        old_val = new_val;

#ifdef OPENCL_PROFILE
        {
            VECTOR_CLASS<cl::Event> events_vector(1);
            status = queue.enqueueNDRangeKernel(plateau_kernel, NullRange, global, local, __null, &events_vector[0]);

            cl::WaitForEvents(events_vector);
            cl::Event& event = events_vector[0];
            cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            cl_ulong total_time = end - start;

            watershed_plateau_kernel_time += total_time;
        }
#else
        status = queue.enqueueNDRangeKernel(plateau_kernel, NullRange, global, local);
#endif
        queue.enqueueReadBuffer(counter, CL_TRUE, 0, sizeof(int), &new_val);
        c++;
    }

#ifdef DEBUG_PRINT
    std::cout << "step 3: " << c << " iterations" << std::endl;
#endif

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

    int n_width = ((width - 1) / new_block_size + 2) * new_block_size;
    int n_height = ((height - 1) / new_block_size + 2) * new_block_size;

    global = cl::NDRange(n_width, n_height);

#ifdef DEBUG_PRINT
    std::cout << "flood kernel invocation params:" << std::endl;
    std::cout << "local: " << local[0] << ", " << local[1] << std::endl;
    std::cout << "global: " << global[0] << ", " << global[1] << std::endl;
#endif

#ifdef OPENCL_PROFILE
    cl::Event last_event;
#endif


#ifdef OPENCL_PROFILE
    watershed_flood_kernel_time = 0;
#endif

    while(old_val != new_val)
    {
        old_val = new_val;

#ifdef OPENCL_PROFILE
        {
            VECTOR_CLASS<cl::Event> events_vector(1);
            status = queue.enqueueNDRangeKernel(flood_kernel, NullRange, global, local, __null, &events_vector[0]);

            cl::WaitForEvents(events_vector);
            cl::Event& event = events_vector[0];
            cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            cl_ulong total_time = end - start;

            watershed_flood_kernel_time += total_time;
        }
#else
        status = queue.enqueueNDRangeKernel(flood_kernel, NullRange, global, local);
#endif

#ifdef OPENCL_PROFILE
        queue.enqueueReadBuffer(counter, CL_TRUE, 0, sizeof(int), &new_val, __null, &last_event);
#else
        queue.enqueueReadBuffer(counter, CL_TRUE, 0, sizeof(int), &new_val);
#endif
        c++;
    }

#ifdef OPENCL_PROFILE
    watershed_descent_kernel_time /= TIME_DIVISOR;
    watershed_increment_kernel_time /= TIME_DIVISOR;
    watershed_minima_kernel_time /= TIME_DIVISOR;
    watershed_plateau_kernel_time /= TIME_DIVISOR;
    watershed_flood_kernel_time /= TIME_DIVISOR;
#endif

#ifdef DEBUG_PRINT
    std::cout << "step 4: " << c << " iterations" << std::endl;
#endif

#ifdef OPENCL_PROFILE
    last_event.wait();

    cl_ulong start = first_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end = last_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    cl_ulong total_time = end - start;

    setLastExecutionTime(total_time/TIME_DIVISOR);
#endif
}














