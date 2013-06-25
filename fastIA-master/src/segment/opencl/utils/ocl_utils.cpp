#include "ocl_utils.h"

#include <iostream>

void oclSimpleInit(cl_device_type type,
                   cl::Context& context, std::vector<cl::Device>& devices)
{
    std::vector<cl::Platform> platforms;

    cl::Platform::get(&platforms);

    if (!platforms.size())
    {
        std::cout << "Platform size 0" << std::endl;
    }

    cl_context_properties properties[] =
    {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};

    context = cl::Context(type, properties);

    devices = context.getInfo<CL_CONTEXT_DEVICES>();

    for(int i = 0; i < devices.size(); ++i){

        cl::Device& device = devices[i];

        std::cout << device.getInfo<CL_DEVICE_NAME>() << std::endl;

    }

}

void oclPrintError(cl::Error &error)
{
    std::cerr << "ERROR: " << error.what() << "(" << error.err() << ")" << std::endl;
}


void ocvMatToOclBuffer(cv::Mat& mat, cl::Buffer& buffer, cl::Context& context, cl::CommandQueue& queue)
{
    size_t data_size = mat.step * mat.size().height;

    buffer = cl::Buffer(context, CL_TRUE, data_size);
    queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, data_size, mat.data);
}

void oclBufferToOcvMat(cv::Mat& mat, cl::Buffer& buffer, int size, cl::Context& context, cl::CommandQueue& queue)
{
    queue.enqueueReadBuffer(buffer, CL_TRUE, 0, size, mat.data);
}
