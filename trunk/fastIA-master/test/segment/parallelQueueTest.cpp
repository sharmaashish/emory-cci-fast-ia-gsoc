#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/test/execution_monitor.hpp>
#include "TestUtils.h"
#include "opencl/utils/ocl_utils.h"

#include <stdio.h>
#include "opencv2/opencv.hpp"

#include "opencl/parallel_queue.h"


/*
 * runtime parameter --catch_system_errors=no  is needed if there is no gpu
 * (opencl internally invokes subprocess that return non-zero value)
 **/

BOOST_AUTO_TEST_CASE(queue_init_test)
{
    std::cout << "parallel queue test hello" << std::endl;

    cl_int err = CL_SUCCESS;

    const int input_size = 512;
    const int output_size = 512;
    const int result_size = 512;

    const int input_size_bytes = input_size * sizeof(int);
    const int output_size_bytes = output_size * sizeof(int);
    const int result_size_bytes = result_size * sizeof(int);

    try
    {
        cl::CommandQueue queue = ProgramCache::getGlobalInstance().getDefaultCommandQueue();
        cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

        initQueueSystem(10);

        int host_in[input_size];
        int host_out[output_size];
        int host_result[result_size];

        std::fill(host_in, host_in + input_size, 1);
        std::fill(host_out, host_out + output_size, 0);

        cl::Buffer device_in(context, CL_TRUE, input_size_bytes);
        queue.enqueueWriteBuffer(device_in, CL_TRUE, 0, input_size_bytes, host_in);

        cl::Buffer device_out(context, CL_TRUE, output_size_bytes);
        queue.enqueueWriteBuffer(device_out, CL_TRUE, 0, output_size_bytes, host_out);

        // initialize queue
        initQueue(device_in, input_size, device_out, output_size);

        cl::Buffer device_result(context, CL_TRUE, result_size_bytes);

        dequeueTest(device_result);

        queue.enqueueReadBuffer(device_result, CL_TRUE, 0, result_size_bytes, host_result);

        for(int i = 0; i < result_size; ++i)
        {
            std::cout << "result[" << i << "] = " << host_result[i] << std::endl;
        }

    }
    catch (cl::Error err)
    {
        oclPrintError(err);
    }

    std::cout << "init queue test finished" << std::endl;
}


BOOST_AUTO_TEST_CASE(queue_sum_test)
{
    std::cout << "parallel queue sum test hello" << std::endl;

    cl_int err = CL_SUCCESS;

    const int input_size = 2224;
    const int output_size = 1024;
    //const int result_size = 512;

    const int iterations = 10;
    const int result_size = 1;


    const int input_size_bytes = input_size * sizeof(int);
    const int output_size_bytes = output_size * sizeof(int);
    const int result_size_bytes = result_size * sizeof(int);

    try
    {
        cl::CommandQueue queue = ProgramCache::getGlobalInstance().getDefaultCommandQueue();
        cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

        initQueueSystem(10);

        int host_in[input_size];
        int host_out[output_size];
        int host_result[result_size];

        std::fill(host_in, host_in + input_size, 1);
        std::fill(host_out, host_out + output_size, 0);

        cl::Buffer device_in(context, CL_TRUE, input_size_bytes);
        queue.enqueueWriteBuffer(device_in, CL_TRUE, 0, input_size_bytes, host_in);

        cl::Buffer device_out(context, CL_TRUE, output_size_bytes);
        queue.enqueueWriteBuffer(device_out, CL_TRUE, 0, output_size_bytes, host_out);

        // initialize queue
        initQueue(device_in, input_size, device_out, output_size);

        cl::Buffer device_result(context, CL_TRUE, result_size_bytes);

        std::cout << "parallel sum..." << std::endl;
        //dequeueTest(device_result);
        sumTest(device_result, iterations);

        queue.enqueueReadBuffer(device_result, CL_TRUE, 0, result_size_bytes, host_result);

        for(int i = 0; i < result_size; ++i)
        {
            std::cout << "result[" << i << "] = " << host_result[i] << std::endl;
        }

    }
    catch (cl::Error err)
    {
        oclPrintError(err);
    }

    std::cout << "sum queue test finished" << std::endl;
}

