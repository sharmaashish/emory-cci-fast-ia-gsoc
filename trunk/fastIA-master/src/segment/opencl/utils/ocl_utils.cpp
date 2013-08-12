#include "ocl_utils.h"

#include <iostream>

void oclSimpleInit(cl_device_type type,
                   cl::Context& context, std::vector<cl::Device>& devices)
{
    std::vector<cl::Platform> platforms;
    cl_int err = CL_SUCCESS;

    cl::Platform::get(&platforms);

    if (!platforms.size())
    {
        std::cout << "Platform size 0" << std::endl;
    }
    else
    {
        std::cout << "Platforms size: " << platforms.size() << std::endl;    
        std::cout << "Platform name: "
                  << platforms[0].getInfo<CL_PLATFORM_NAME>() << std::endl;
    }

    cl_context_properties properties[] =
    {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};

    context = cl::Context(type, properties, NULL, NULL, &err);
    
    //std::cout << (err == CL_SUCCESS ? "true" : "false") << std::endl;

    devices = context.getInfo<CL_CONTEXT_DEVICES>();

    for(int i = 0; i < devices.size(); ++i){

        cl::Device& device = devices[i];

        std::cout << device.getInfo<CL_DEVICE_NAME>() << std::endl;

    }

}

void oclPrintError(cl::Error &error)
{
    std::cerr << "ERROR: " << error.what()
              << "(" << error.err() << ")" << std::endl;
}

cl::Buffer ocvMatToOclBuffer(cv::Mat& mat, cl::CommandQueue& queue)
{
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

    int size_in_bytes = mat.rows * mat.step;

    cl::Buffer buffer(context, CL_TRUE, size_in_bytes);
    queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, size_in_bytes, mat.data);

    return buffer;
}

void ocvMatToOclBuffer(cv::Mat& mat, cl::Buffer& buffer,
                       cl::Context& context, cl::CommandQueue& queue)
{
    size_t data_size = mat.step * mat.size().height;

    buffer = cl::Buffer(context, CL_TRUE, data_size);
    queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, data_size, mat.data);
}

void oclBufferToOcvMat(cv::Mat& mat, cl::Buffer& buffer,
                       int size, cl::CommandQueue& queue)
{
    queue.enqueueReadBuffer(buffer, CL_TRUE, 0, size, mat.data);
}

void oclBuferToOcvMat(cv::Mat& mat, cl::Buffer buffer,
                      cl::CommandQueue queue)
{
    queue.enqueueReadBuffer(buffer, CL_TRUE,
                            0, mat.step * mat.size().height, mat.data);
}


#ifdef OPENCL_PROFILE

static float executionTime;

float getLastExecutionTime()
{
    return executionTime;
}

void setLastExecutionTime(float time)
{
    executionTime = time;
}

bool checkProfilingSupport(cl::CommandQueue& queue)
{
    cl_command_queue_properties queue_properties
            = queue.getInfo<CL_QUEUE_PROPERTIES>();

    return (queue_properties & CL_QUEUE_PROFILING_ENABLE) != 0;
}

#endif
